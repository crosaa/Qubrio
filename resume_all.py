"""Resume an interrupted batch from a previous log directory."""
from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from api_client import PROVIDERS
from incremental_compiler import generate_transition
from verify import Hardware, apply_aod_round, infer_n_qubits, verify_output
from log_parser import load_checkpoint, parse_job_logs
from run_all import ThreadLogger, _thread_aware_print, discover_batch_datasets

import builtins
builtins.print = _thread_aware_print


def resume_dataset(
    label: str,
    cz_stages: List,
    log_path: Path,
    result_path: Path,
    api_key: str,
    inner_max_retries: int,
    outer_max_retries: int,
    parsed_state: tuple,
    provider: str = "deepseek",
    model: str | None = None,
    effort: str = "medium",
) -> Dict[str, Any]:
    """Resume and compile one dataset. Runs inside a worker thread."""

    initial_layout, schedule, completed_stages_count, current_layout = parsed_state

    with ThreadLogger(label, log_path):
        try:
            _thread_aware_print("=" * 60)
            _thread_aware_print(f"Dataset : {label} (RESUMING)")
            _thread_aware_print(f"CZ stages total: {len(cz_stages)}")
            _thread_aware_print(f"Completed stages: {completed_stages_count}")
            _thread_aware_print(f"Log file : {log_path}")
            _thread_aware_print("=" * 60)

            from api_client import LLMClient
            from math import ceil, sqrt
            from incremental_compiler import _set_llm_log_path

            _set_llm_log_path(log_path.parent / "llm_log.txt")
            client = LLMClient(api_key, provider=provider, model=model, effort=effort)

            n_qubits = infer_n_qubits(cz_stages)
            side = ceil(sqrt(n_qubits))
            hw = Hardware(n_qubits=n_qubits, width=2 * side, height=side)

            _thread_aware_print("=" * 60)
            _thread_aware_print(f"Incremental Quantum Compiler (RESUMED)")
            _thread_aware_print(f"N_QUBITS={n_qubits}, WIDTH={hw.width}, HEIGHT={hw.height}")
            _thread_aware_print(f"Inner max retries: {inner_max_retries}, Outer max retries: {outer_max_retries}")
            _thread_aware_print("=" * 60)
            
            # Splice the real cz_stages back into the recovered schedule.
            for i, stage_idx in enumerate(range(completed_stages_count)):
                 if 2 * stage_idx + 1 < len(schedule):
                      schedule[2 * stage_idx + 1]["cz_stage"] = cz_stages[stage_idx]

            for stage_idx in range(completed_stages_count, len(cz_stages)):
                cz_stage = cz_stages[stage_idx]
                previous_failures = []

                for retry_attempt in range(outer_max_retries):
                    try:
                        aod_groups, target_layout, _unopt = generate_transition(
                            stage_idx - 1,
                            current_layout,
                            cz_stages,
                            hw,
                            client,
                            inner_max_retries,
                            verbose=True,
                            previous_failures=previous_failures
                        )

                        formatted_aod_groups = []
                        for rnd in aod_groups:
                            formatted_rnd = []
                            for move in rnd:
                                q, (x1, y1), (x2, y2) = move
                                formatted_rnd.append([q, [x1, y1], [x2, y2]])
                            formatted_aod_groups.append(formatted_rnd)

                        formatted_target_layout = {str(q): [pos[0], pos[1]] for q, pos in sorted(target_layout.items())}

                        schedule.append({
                            "target_layout": formatted_target_layout,
                            "aod_move_groups": formatted_aod_groups
                        })

                        schedule.append({"cz_stage": cz_stage})

                        for rnd in aod_groups:
                            current_layout = apply_aod_round(current_layout, rnd)

                        _thread_aware_print(f"[OK] Transition to CZ Stage {stage_idx + 1} completed")
                        break

                    except RuntimeError as e:
                        error_msg = str(e)
                        previous_failures.append((retry_attempt, "(Code from this attempt was discarded)", error_msg))

                        _thread_aware_print(f"[FAIL] Transition verification failed (attempt {retry_attempt + 1}/{outer_max_retries})")
                        _thread_aware_print(f"[ERROR FEEDBACK] {error_msg}")

                        # Roll back to the last confirmed stage (AOD then CZ → 2 entries each).
                        confirmed_len = stage_idx * 2
                        del schedule[confirmed_len:]

                        current_layout = dict(initial_layout)
                        for i in range(0, confirmed_len, 2):
                            for rnd_formatted in schedule[i]["aod_move_groups"]:
                                rnd = [(m[0], tuple(m[1]), tuple(m[2])) for m in rnd_formatted]
                                current_layout = apply_aod_round(current_layout, rnd)

                        if retry_attempt == outer_max_retries - 1:
                            raise RuntimeError(f"Failed to generate valid transition to CZ Stage {stage_idx + 1} after {outer_max_retries} outer attempts")

            formatted_initial_layout = {str(q): [pos[0], pos[1]] for q, pos in sorted(initial_layout.items())}

            result = {
                "n_qubits": n_qubits,
                "width": hw.width,
                "height": hw.height,
                "initial_layout": formatted_initial_layout,
                "schedule": schedule
            }

            _thread_aware_print("\n" + "=" * 60)
            _thread_aware_print("[SUCCESS] Incremental compilation complete")
            _thread_aware_print("=" * 60)

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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Resume interrupted quantum compiler runs from a previous log directory.\nWill output logs into the exact same folder continuing exactly where it left off.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume-log-dir", type=str, required=True,
        help="Path to the interrupted log directory to resume, e.g. logs/20260303_005438.",
    )
    parser.add_argument(
        "--test-dir", type=str, required=True,
        help="Directory containing the original benchmark *.txt files. Required.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel threads (default: one per dataset).",
    )
    parser.add_argument(
        "--inner-retries", type=int, default=13,
        help="Inner retry limit per LLM call (default: 13).",
    )
    parser.add_argument(
        "--outer-retries", type=int, default=4,
        help="Outer retry limit per transition (default: 4).",
    )
    parser.add_argument(
        "--provider", type=str, default="deepseek",
        choices=list(PROVIDERS.keys()),
        help="LLM backbone: deepseek (default), openai, or qwen.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Override LLM model name (e.g. 'gpt-5.4', 'qwen/qwen3.5-397b-a17b'). "
            "Defaults to the provider's default."
        ),
    )
    parser.add_argument(
        "--effort", type=str, default="high",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for thinking-mode backbones (default: high).",
    )
    args = parser.parse_args()

    provider = args.provider
    env_key = PROVIDERS[provider]["env_key"]
    api_key = os.environ.get(env_key)
    if not api_key:
        print(f"[ERROR] Environment variable {env_key} is not set.")
        sys.exit(1)

    script_dir = Path(__file__).parent
    resume_root = script_dir / args.resume_log_dir
    test_dir = script_dir / args.test_dir

    if not resume_root.is_dir():
        print(f"[ERROR] Resume log directory not found: {resume_root}")
        sys.exit(1)

    datasets = discover_batch_datasets(test_dir)
    if not datasets:
        print("[ERROR] No valid datasets found in test-dir.")
        sys.exit(1)

    print("=" * 70)
    print(f"Resuming Multi-threaded Quantum Compiler")
    print(f"Resume Root : {resume_root}")
    print(f"Workers     : {args.workers or len(datasets)}")
    print("=" * 70)

    jobs = []
    skipped_completed = []
    skipped_no_logs = []
    skipped_parsing_err = []
    loaded_from_checkpoint = []
    loaded_from_logs = []

    for label, cz_stages, txt_path, group in datasets:
        circuit = txt_path.stem
        job_dir = resume_root / group / circuit
        log_path = job_dir / "output.txt"
        result_path = job_dir / "result.json"
        checkpoint_path = job_dir / "checkpoint.json"

        # Check if already fully completed
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8")
            if "[OK] Compilation successful!" in text:
                skipped_completed.append(label)
                continue

        # Prefer structured checkpoint.json; fall back to log parsing for
        # older runs that predate the checkpoint feature.
        parsed_state = None
        if checkpoint_path.exists():
            try:
                parsed_state = load_checkpoint(job_dir)
                loaded_from_checkpoint.append((label, parsed_state[2]))
            except Exception as e:
                print(f"[{label}] checkpoint load failed ({e}); "
                      f"falling back to log parsing")

        if parsed_state is None:
            # Log-parsing fallback requires both output.txt and llm_log.txt.
            if not log_path.exists() or not (job_dir / "llm_log.txt").exists():
                skipped_no_logs.append(label)
                continue
            try:
                parsed_state = parse_job_logs(job_dir)
                loaded_from_logs.append((label, parsed_state[2]))
            except Exception as e:
                skipped_parsing_err.append((label, str(e)))
                continue

        # Ready to resume.
        # Note: resume appends to existing logs; ThreadLogger opens in "w"
        # by default, so main() monkey-patches __enter__ to use "a" below.
        jobs.append((label, cz_stages, log_path, result_path, parsed_state))

    # Monkey patch thread logger to append instead of overwriting "w" mode
    def threadlogger_enter(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # CHANGE w -> a
            self._file = open(self.log_path, "a", encoding="utf-8")
        except Exception as exc:
            self._real_stderr.write(
                f"[{self.label}] WARNING: could not open log file: {exc}\n"
            )
        ThreadLogger._local.logger = self
        return self
        
    ThreadLogger.__enter__ = threadlogger_enter

    if loaded_from_checkpoint:
        print(f"[RESUME] {len(loaded_from_checkpoint)} datasets loaded from checkpoint.json:")
        for l, n in loaded_from_checkpoint:
            print(f"  - {l}: {n} stages completed")
    if loaded_from_logs:
        print(f"[RESUME] {len(loaded_from_logs)} datasets loaded from log parsing (legacy):")
        for l, n in loaded_from_logs:
            print(f"  - {l}: {n} stages completed")
    if skipped_completed:
        print(f"[SKIP] {len(skipped_completed)} datasets already fully complete.")
    if skipped_no_logs:
        print(f"[SKIP] {len(skipped_no_logs)} datasets had no previous logs to resume.")
    if skipped_parsing_err:
        print(f"[SKIP] {len(skipped_parsing_err)} datasets failed to parse for resume state:")
        for l, err in skipped_parsing_err:
            print(f"  - {l}: {err}")
    print("")

    if not jobs:
        print("[INFO] No viable tasks found to resume.")
        sys.exit(0)

    max_workers = args.workers or len(jobs)
    futures = {}
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for label, cz_stages, log_path, result_path, parsed_state in jobs:
            future = executor.submit(
                resume_dataset,
                label, cz_stages, log_path, result_path,
                api_key,
                args.inner_retries,
                args.outer_retries,
                parsed_state,
                provider,
                args.model,
                args.effort,
            )
            futures[future] = label

        for future in as_completed(futures):
            all_results.append(future.result())

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<32} {'Status':<8} Detail")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x["label"]):
        ok = r["success"]
        status = "OK    " if ok else "FAIL  "
        detail = r["log_path"] if ok else r["error"]
        print(f"{r['label']:<32} {status}  {detail}")
    print("=" * 70)

    n_ok = sum(1 for r in all_results if r["success"])
    print(f"\nResult: {n_ok}/{len(all_results)} succeeded.")
    if n_ok < len(all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
