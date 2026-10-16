#!/usr/bin/env python3
"""End-to-end driver: compile (via run_all.py) → aggregate → simulate.

Output performance JSONs contain ``total_duration``, ``final_fidelity``,
``movement_fidelity``, and ``coherence_fidelity`` per circuit. Pass
``--skip-compile --log-root logs/<timestamp>`` to reuse an existing run.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from aggregate_results import aggregate_results
from performance_simulation import calculate_transition_performance


_TIMESTAMP_RE = re.compile(r'^\d{8}_\d{6}$')


def find_latest_log_dir(log_base: Path) -> Path:
    """Return the newest ``YYYYMMDD_HHMMSS``-named subdir.

    Filtering by the timestamp pattern keeps ad-hoc subdirs from
    out-sorting real run dirs.
    """
    subdirs = sorted(
        [d for d in log_base.iterdir() if d.is_dir() and _TIMESTAMP_RE.match(d.name)],
        key=lambda d: d.name,
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No timestamped log directories found in {log_base}")
    return subdirs[0]


def run_compile(args):
    """Invoke run_all.py with the relevant CLI flags."""
    cmd = [
        sys.executable, "run_all.py",
        "--test-dir", args.test_dir,
        "--workers", str(args.workers),
        "--inner-retries", str(args.inner_retries),
        "--outer-retries", str(args.outer_retries),
        "--provider", args.provider,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.effort:
        cmd += ["--effort", args.effort]
    if args.groups:
        cmd += ["--groups"] + args.groups
    if args.instances:
        cmd += ["--instances"] + args.instances
    if args.skip_ok_in:
        cmd += ["--skip-ok-in", args.skip_ok_in]

    print("=" * 70)
    print(f"[STEP 1] Compiling circuits")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)
    subprocess.run(cmd, check=True)


def run_aggregate_and_perf(log_root: Path, groups: list, out_dir: Path):
    """Aggregate per-circuit results and compute simulator metrics
    for both the optimized and unoptimized schedules."""
    out_dir.mkdir(parents=True, exist_ok=True)

    perf_files = {}  # label -> perf json path

    for group in groups:
        group_dir = log_root / group
        if not group_dir.is_dir():
            print(f"[WARN] Group dir not found: {group_dir}, skipping.")
            continue

        for suffix, tag in [("result.json", "optimized"), ("result_unoptimized.json", "unoptimized")]:
            label = f"{group}_{tag}"
            agg_file = out_dir / f"{label}_aggregated.json"

            # --- Aggregate ---
            print(f"\n[STEP 2] Aggregating {tag} results for {group}")
            agg_data = aggregate_results(group_dir, output_file=agg_file, suffix=suffix)

            if not agg_data["results"]:
                print(f"  No results found for {label}, skipping.")
                continue

            # Infer n_qubits from group name (e.g. q20_regular5 -> 20).
            m = re.match(r"q(\d+)", group)
            default_n_qubits = int(m.group(1)) if m else None

            print(f"[STEP 3] Computing performance for {label}")
            all_metrics = {}
            for circuit_id, item in agg_data["results"].items():
                schedule = item.get("schedule", [])
                n_qubits = item.get("n_qubits", default_n_qubits)
                if n_qubits is None:
                    print(f"  [ERROR] Cannot determine n_qubits for {circuit_id}, skipping.")
                    continue

                all_move_groups = []
                cz_stages = []
                for stage in schedule:
                    if "aod_move_groups" in stage:
                        all_move_groups.extend(stage["aod_move_groups"])
                    if "cz_stage" in stage:
                        cz_stages.append(stage["cz_stage"])

                metrics = calculate_transition_performance(
                    all_move_groups, n_qubits, cz_stages=cz_stages
                )
                all_metrics[circuit_id] = metrics
                print(f"  {circuit_id}: duration={metrics['total_duration']:.1f}μs, fidelity={metrics['final_fidelity']:.6f}")

            perf_file = out_dir / f"{label}_performance.json"
            with open(perf_file, "w") as f:
                json.dump(all_metrics, f, indent=4)
            print(f"  Saved performance: {perf_file}")

            perf_files[label] = str(perf_file)

    return perf_files


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: compile → aggregate → performance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Compile flags forwarded to run_all.py.
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Directory containing benchmark *.txt files (required unless --skip-compile).")
    parser.add_argument("--groups", type=str, nargs="+", default=None)
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--inner-retries", type=int, default=10,
                        help="Inner retry limit per LLM call (default: 10).")
    parser.add_argument("--outer-retries", type=int, default=5,
                        help="Outer retry limit per transition (default: 5).")
    parser.add_argument("--provider", type=str, default="deepseek",
                        choices=["deepseek", "openai", "qwen"],
                        help="LLM backbone: deepseek (default), openai, or qwen.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model name (e.g. 'gpt-5.4', "
                             "'qwen/qwen3.5-397b-a17b'). Defaults to the provider's default.")
    parser.add_argument("--effort", type=str, default="high",
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort for thinking-mode backbones (default: high).")
    parser.add_argument("--skip-ok-in", type=str, default=None)

    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip the compile step and reuse existing logs.")
    parser.add_argument("--log-root", type=str, default=None,
                        help="Explicit log root. Defaults to the latest dir in logs/.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (defaults to <log-root>/pipeline_output).")

    args = parser.parse_args()
    script_dir = Path(__file__).parent

    if not args.skip_compile:
        if not args.test_dir:
            parser.error("--test-dir is required when not using --skip-compile")
        run_compile(args)

    if args.log_root:
        log_root = script_dir / args.log_root
    else:
        log_root = find_latest_log_dir(script_dir / "logs")
    print(f"\n[INFO] Using log root: {log_root}")

    if args.groups:
        groups = args.groups
    else:
        groups = sorted([
            d.name for d in log_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    print(f"[INFO] Groups: {groups}")

    out_dir = Path(args.out_dir) if args.out_dir else log_root / "pipeline_output"
    perf_files = run_aggregate_and_perf(log_root, groups, out_dir)

    print("\n" + "=" * 70)
    print("[DONE] Pipeline complete!")
    print(f"  Performance JSON files: {list(perf_files.values())}")
    print(f"  Results dir: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
