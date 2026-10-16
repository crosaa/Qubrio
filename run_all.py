"""Multi-threaded compiler runner.

Default batch mode walks ``<test-dir>/**/*.txt`` (stage-report format,
parsing ``cz_stages`` from ``Gates:`` lines) and writes logs to
``logs/<timestamp>/<group>/<circuit>/``. ``--legacy`` switches to the
old ``data/q_*`` Python-assignment format.

Set the API key matching ``--provider``: DEEPSEEK_API_KEY,
OPENAI_API_KEY, or OPENROUTER_API_KEY (for ``qwen``).
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from api_client import PROVIDERS
from incremental_compiler import compile_incremental
from verify import verify_output


# ── Thread-safe per-thread logger ───────────────────────────────────────────

_TERM_LOCK = threading.Lock()


class ThreadLogger:
    """Capture a single thread's prints to a UTF-8 log file and echo
    label-prefixed lines to the shared terminal."""
    _local = threading.local()

    def __init__(self, label: str, log_path: Path):
        self.label = label
        self.log_path = log_path
        self._real_stdout = sys.__stdout__
        self._real_stderr = sys.__stderr__
        self._term_lock = _TERM_LOCK
        self._file: Optional[io.TextIOWrapper] = None

    def __enter__(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_path, "w", encoding="utf-8")
        except Exception as exc:
            self._real_stderr.write(
                f"[{self.label}] WARNING: could not open log file: {exc}\n"
            )
        ThreadLogger._local.logger = self
        return self

    def __exit__(self, *_):
        ThreadLogger._local.logger = None
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass

    def write(self, data: str):
        # File copy is UTF-8 so any character is preserved.
        if self._file and data:
            try:
                self._file.write(data)
                self._file.flush()
            except Exception:
                pass
        # Skip pure-whitespace writes so many threads don't flood the terminal.
        stripped = data.strip() if data else ""
        if not stripped:
            return
        lines = data.rstrip("\n").splitlines()
        prefixed = "\n".join(
            f"[{self.label}] {ln}" if ln.strip() else ""
            for ln in lines
        ) + "\n"
        with self._term_lock:
            try:
                self._real_stdout.write(prefixed)
                self._real_stdout.flush()
            except UnicodeEncodeError:
                # cp936/GBK consoles can't encode arbitrary unicode; fall
                # back to a replacement-encoded copy so a logging glitch
                # never kills the worker. The file mirror has the original.
                enc = getattr(self._real_stdout, "encoding", None) or "ascii"
                safe = prefixed.encode(enc, errors="replace").decode(enc, errors="replace")
                try:
                    self._real_stdout.write(safe)
                    self._real_stdout.flush()
                except Exception:
                    pass
            except Exception:
                # Never let any other terminal-side error take down the loop.
                pass

    def flush(self):
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass

# ── Thread-aware print() and direct (non-routed) printer ────────────────────

_builtin_print = print


def _original_print(*args, sep=" ", end="\n", file=None, flush=False):
    """Direct print that bypasses the thread router and survives a
    Windows console that can't encode the payload (cp936/GBK)."""
    try:
        _builtin_print(*args, sep=sep, end=end, file=file, flush=flush)
    except UnicodeEncodeError:
        target = file if file is not None else sys.__stdout__
        enc = getattr(target, "encoding", None) or "ascii"
        text = sep.join(str(a) for a in args) + end
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        try:
            target.write(safe)
            if flush:
                target.flush()
        except Exception:
            pass


def _thread_aware_print(*args, sep=" ", end="\n", file=None, flush=False):
    logger: Optional[ThreadLogger] = getattr(ThreadLogger._local, "logger", None)
    if logger is not None and file is None:
        # Build the full text exactly as print() would, including custom end
        text = sep.join(str(a) for a in args) + end
        logger.write(text)
        if flush:
            logger.flush()
    else:
        _original_print(*args, sep=sep, end=end, file=file, flush=flush)

import builtins
builtins.print = _thread_aware_print


# ──────────────────────────────────────────────────────────────────────────────
# Parser: stage-report format  (<test-dir>/**/*.txt)
# ──────────────────────────────────────────────────────────────────────────────

def parse_stage_report(txt_path: Path) -> List[List[Tuple[int, int]]]:
    """Parse ``Stage N:\n  Gates: [...]`` blocks, ordered by stage number."""
    text = txt_path.read_text(encoding="utf-8")

    stage_pattern = re.compile(
        r"Stage\s+(\d+):\s*\n(?:.*\n)*?\s+Gates:\s*(\[.*?\])",
        re.MULTILINE,
    )
    matches = stage_pattern.findall(text)

    if not matches:
        raise ValueError(f"No Stage/Gates data found in {txt_path}")

    matches.sort(key=lambda m: int(m[0]))

    cz_stages = []
    for stage_num, gates_str in matches:
        gates = ast.literal_eval(gates_str)
        if not isinstance(gates, list):
            raise ValueError(f"Gates in stage {stage_num} is not a list")
        cz_stages.append([(int(a), int(b)) for a, b in gates])

    return cz_stages


# ── Legacy Python-assignment dataset format ─────────────────────────────────

def parse_python_assignment(txt_path: Path) -> List:
    source = txt_path.read_text(encoding="utf-8")
    ns: Dict[str, Any] = {}
    exec(compile(source, str(txt_path), "exec"), ns)  # noqa: S102
    if "cz_stages" not in ns:
        raise ValueError(f"'cz_stages' variable not found in {txt_path}")
    return ns["cz_stages"]


# ── Dataset discovery ───────────────────────────────────────────────────────

def discover_batch_datasets(
    test_dir: Path,
) -> List[Tuple[str, List, Path, str]]:
    """Walk ``test_dir`` and yield (label, cz_stages, txt_path, group).

    Accepts both layouts: ``benchmark/<group>/<circuit>.txt`` or a flat
    ``benchmark/*.txt``.
    """
    datasets = []
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    def _numeric_key(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    def _add_from_dir(d: Path, group_name: str):
        for txt_path in sorted(d.glob("*.txt"), key=_numeric_key):
            label = f"{group_name}/{txt_path.stem}"
            try:
                cz_stages = parse_stage_report(txt_path)
            except Exception as exc:
                _original_print(f"[WARN] Skipping {txt_path}: {exc}")
                continue
            datasets.append((label, cz_stages, txt_path, group_name))

    # 1. Check if test_dir itself has i*.txt files
    direct_txts = list(test_dir.glob("*.txt"))
    if direct_txts:
        _add_from_dir(test_dir, test_dir.name)

    # 2. Check subdirectories for group folders
    for group_dir in sorted(test_dir.iterdir()):
        if group_dir.is_dir():
            _add_from_dir(group_dir, group_dir.name)

    return datasets


def discover_legacy_datasets(
    data_dir: Path,
) -> List[Tuple[str, List, Path, str]]:
    """
    Walk data_dir for q_* folders containing q_*.txt (Python assignment format).
    Returns list of (label, cz_stages, txt_path, group) tuples.
    """
    datasets = []
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("q_"):
            continue
        txt_files = list(folder.glob("q_*.txt"))
        if not txt_files:
            _original_print(f"[WARN] No q_*.txt in {folder}, skipping.")
            continue
        txt_path = txt_files[0]
        try:
            cz_stages = parse_python_assignment(txt_path)
        except Exception as exc:
            _original_print(f"[WARN] Skipping {txt_path}: {exc}")
            continue
        datasets.append((folder.name, cz_stages, txt_path, folder.name))

    return datasets


# ── Per-dataset worker ──────────────────────────────────────────────────────

def run_dataset(
    label: str,
    cz_stages: List,
    log_path: Path,
    result_path: Path,
    api_key: str,
    inner_max_retries: int,
    outer_max_retries: int,
    provider: str = "deepseek",
    model: Optional[str] = None,
    effort: str = "medium",
) -> Dict[str, Any]:
    """Compile one dataset. Runs inside a thread."""

    with ThreadLogger(label, log_path):
        try:
            _thread_aware_print("=" * 60)
            _thread_aware_print(f"Dataset : {label}")
            _thread_aware_print(f"CZ stages: {len(cz_stages)}")
            _thread_aware_print(f"Log file : {log_path}")
            _thread_aware_print("=" * 60)

            result, unoptimized_result = compile_incremental(
                cz_stages,
                api_key,
                provider=provider,
                inner_max_retries=inner_max_retries,
                outer_max_retries=outer_max_retries,
                verbose=True,
                llm_log_path=log_path.parent / "llm_log.txt",
                model=model,
                effort=effort,
            )

            # End-to-end verification
            _thread_aware_print("\n" + "=" * 60)
            _thread_aware_print("Final End-to-End Verification")
            _thread_aware_print("=" * 60)
            verify_ok = False
            verify_err = None
            try:
                verify_output(cz_stages, result, require_exact_cz_match=True)
                _thread_aware_print("[VERIFY] All constraints satisfied")
                _thread_aware_print("[OK] Compilation successful!")
                verify_ok = True
            except ValueError as ve:
                verify_err = str(ve)
                _thread_aware_print(f"[FAIL] Verification failed: {ve}")

            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            _thread_aware_print(f"\n[SAVED] {result_path}")

            unopt_result_path = result_path.parent / "result_unoptimized.json"
            unopt_result_path.write_text(json.dumps(unoptimized_result, indent=2), encoding="utf-8")
            _thread_aware_print(f"[SAVED] {unopt_result_path}")

            return {
                "label": label,
                "success": verify_ok,
                "error": verify_err,
                "log_path": str(log_path),
                "result_path": str(result_path),
            }

        except Exception as exc:
            tb = traceback.format_exc()
            _thread_aware_print(f"\n[FATAL ERROR] {exc}")
            _thread_aware_print(tb)
            return {
                "label": label,
                "success": False,
                "error": str(exc),
                "log_path": str(log_path),
                "result_path": None,
            }


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Multi-threaded quantum compiler.\n"
            "Default: batch mode over benchmark/*.txt\n"
            "Use --legacy for old data/q_* format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-dir", type=str, required=True,
        help="Directory containing benchmark *.txt files. Required.",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy mode: scan --data-dir for q_* folders with q_*.txt files.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root data directory for legacy mode (default: data/).",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Root directory for log output (default: logs/).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel threads (default: one per dataset).",
    )
    parser.add_argument(
        "--inner-retries", type=int, default=10,
        help="Inner retry limit per LLM call (default: 10).",
    )
    parser.add_argument(
        "--outer-retries", type=int, default=5,
        help="Outer retry limit per transition (default: 5).",
    )
    parser.add_argument(
        "--provider", type=str, default="deepseek",
        choices=list(PROVIDERS.keys()),
        help="LLM backbone: deepseek (default), openai, or qwen.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Override LLM model name (e.g. 'gpt-5.4', 'deepseek-reasoner', "
            "'qwen/qwen3.5-397b-a17b'). Defaults to the provider's default."
        ),
    )
    parser.add_argument(
        "--effort", type=str, default="high",
        choices=["minimal", "low", "medium", "high"],
        help=(
            "Reasoning effort for thinking-mode backbones (default: high, "
            "matching the paper's run configuration). "
            "Drop to medium/low if you need faster turnaround at the cost of "
            "schedule quality."
        ),
    )
    parser.add_argument(
        "--max-stages", type=int, default=None,
        help=(
            "Truncate each circuit to its first N CZ stages (smoke testing). "
            "Default: run the full circuit."
        ),
    )
    parser.add_argument(
        "--groups", type=str, nargs="+", default=None,
        help="Only run these group folders (e.g. q20_regular5 q20_regular6).",
    )
    parser.add_argument(
        "--instances", type=str, nargs="+", default=None,
        help="Only run these instances (e.g. i0 i1 i2).",
    )
    parser.add_argument(
        "--skip-ok-in", type=str, default=None,
        metavar="LOG_ROOT",
        help=(
            "Skip datasets that already succeeded in a previous run. "
            "Provide the log root directory to check, e.g. logs/20260218_140931. "
            "A dataset is skipped if its output.txt contains '[OK] Compilation successful!'."
        ),
    )
    args = parser.parse_args()

    provider = args.provider
    env_key = PROVIDERS[provider]["env_key"]
    api_key = os.environ.get(env_key)
    if not api_key:
        _original_print(
            f"[ERROR] Environment variable {env_key} is not set.\n"
            f"  PowerShell : $env:{env_key} = 'sk-...'\n"
            f"  CMD        : set {env_key}=sk-...\n"
            f"  Or add it to .env file in the project root."
        )
        sys.exit(1)

    script_dir = Path(__file__).parent

    if args.legacy:
        data_dir = script_dir / args.data_dir
        datasets = discover_legacy_datasets(data_dir)
        mode_label = f"LEGACY  ({data_dir})"
    else:
        test_dir = script_dir / args.test_dir
        datasets = discover_batch_datasets(test_dir)
        mode_label = f"BATCH   ({test_dir})"

    if args.groups:
        datasets = [d for d in datasets if d[3] in args.groups]
    if args.instances:
        datasets = [d for d in datasets if d[2].stem in args.instances]

    if args.max_stages is not None and args.max_stages > 0:
        datasets = [
            (label, stages[: args.max_stages], txt_path, group)
            for (label, stages, txt_path, group) in datasets
        ]

    if not datasets:
        _original_print("[ERROR] No valid datasets found.")
        sys.exit(1)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = script_dir / args.log_dir / run_ts

    effective_model = args.model or PROVIDERS[provider]["default_model"]

    _original_print("=" * 70)
    _original_print(f"Multi-threaded Quantum Compiler")
    _original_print(f"Provider : {provider} ({effective_model}, effort={args.effort})")
    _original_print(f"Mode     : {mode_label}")
    _original_print(f"Datasets : {len(datasets)}")
    _original_print(f"Log root : {log_root}")
    _original_print(f"Workers  : {args.workers or len(datasets)}")
    _original_print(f"Retries  : inner={args.inner_retries}  outer={args.outer_retries}")
    _original_print("=" * 70)
    for label, stages, txt_path, group in datasets:
        _original_print(f"  {label:<30s}  {len(stages)} stages  ←  {txt_path.name}")
    _original_print("=" * 70 + "\n")

    # Layout: logs/<run_ts>/<group>/<circuit>/{output.txt, result.json}
    skip_root: Optional[Path] = None
    if args.skip_ok_in:
        skip_root = script_dir / args.skip_ok_in
        if not skip_root.is_dir():
            _original_print(f"[WARN] --skip-ok-in path not found: {skip_root}")
            skip_root = None

    def _already_succeeded(group: str, circuit: str) -> bool:
        if skip_root is None:
            return False
        prev_log = skip_root / group / circuit / "output.txt"
        if not prev_log.exists():
            return False
        try:
            return "[OK] Compilation successful!" in prev_log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False

    jobs = []
    skipped = []
    for label, cz_stages, txt_path, group in datasets:
        circuit = txt_path.stem          # e.g. "i0"
        if _already_succeeded(group, circuit):
            skipped.append(label)
            continue
        job_dir = log_root / group / circuit
        log_path = job_dir / "output.txt"
        result_path = job_dir / "result.json"
        jobs.append((label, cz_stages, log_path, result_path))

    if skipped:
        _original_print(f"[SKIP] {len(skipped)} datasets already succeeded (--skip-ok-in):")
        for s in skipped:
            _original_print(f"  + {s}")
        _original_print("")

    if not jobs:
        _original_print("[INFO] All datasets already succeeded. Nothing to run.")
        sys.exit(0)

    max_workers = args.workers or len(jobs)
    futures = {}
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for label, cz_stages, log_path, result_path in jobs:
            future = executor.submit(
                run_dataset,
                label, cz_stages, log_path, result_path,
                api_key,
                args.inner_retries,
                args.outer_retries,
                provider,
                args.model,
                args.effort,
            )
            futures[future] = label

        for future in as_completed(futures):
            all_results.append(future.result())

    _original_print("\n" + "=" * 70)
    _original_print("SUMMARY")
    _original_print("=" * 70)
    _original_print(f"{'Dataset':<32} {'Status':<8} Detail")
    _original_print("-" * 70)
    for r in sorted(all_results, key=lambda x: x["label"]):
        ok = r["success"]
        status = "OK    " if ok else "FAIL  "
        detail = r["log_path"] if ok else r["error"]
        _original_print(f"{r['label']:<32} {status}  {detail}")
    _original_print("=" * 70)

    n_ok = sum(1 for r in all_results if r["success"])
    _original_print(f"\nResult: {n_ok}/{len(all_results)} succeeded.")
    _original_print(f"Logs saved under: {log_root}\n")

    if n_ok < len(all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
