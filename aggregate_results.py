"""Combine per-circuit ``result.json`` files into a single grouped JSON."""

import json
from pathlib import Path


def get_depth(data):
    """Maximum nesting depth of ``data`` (primitives = 0, ``[]`` = 1)."""
    if isinstance(data, list):
        if not data:
            return 1
        return 1 + max(get_depth(x) for x in data)
    return 0


def format_data(data, indent=2):
    """JSON-like formatter that keeps short lists on one line.

    Lists of depth ≤ 3 (coords / moves / per-round move groups) stay
    inline; deeper lists and dicts expand multi-line. Result is human
    readable while keeping AOD schedules compact.
    """
    if isinstance(data, list):
        depth = get_depth(data)
        if depth <= 3:
            items = [format_data(x, indent) for x in data]
            return "[" + ", ".join(items) + "]"
        items = [format_data(x, indent) for x in data]
        inner = ",\n".join(items)
        inner_indented = "\n".join(" " * indent + line for line in inner.split('\n'))
        return "[\n" + inner_indented + "\n]"

    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            key_s = json.dumps(k)
            val_s = format_data(v, indent)
            items.append(f"{key_s}: {val_s}")
        inner = ",\n".join(items)
        inner_indented = "\n".join(" " * indent + line for line in inner.split('\n'))
        return "{\n" + inner_indented + "\n}"

    return json.dumps(data)

def aggregate_results(logs_root, output_file=None, suffix="result.json"):
    """Aggregate ``logs_root/<circuit>/<suffix>`` into one ``{group, results}`` dict.

    Pass ``suffix="result_unoptimized.json"`` to aggregate the
    pre-optimization schedules. If ``output_file`` is given, the
    aggregated dict is also written there using the compact formatter.
    """
    root_path = Path(logs_root)
    if not root_path.exists():
        print(f"Error: Directory {logs_root} does not exist.")
        return {"group": "", "results": {}}

    aggregated_data = {
        "group": root_path.name,
        "results": {}
    }

    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()], key=lambda x: x.name)

    for item in subdirs:
        circuit_name = item.name
        result_file = item / suffix

        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    aggregated_data["results"][circuit_name] = data
                print(f"  Loaded {circuit_name}/{suffix}")
            except Exception as e:
                print(f"  Error reading {result_file}: {e}")
        else:
            print(f"  [WARN] Missing {circuit_name}/{suffix}")

    if output_file is not None:
        try:
            output_str = format_data(aggregated_data, indent=2)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"  Saved aggregated ({len(aggregated_data['results'])} circuits): {output_file}")
        except Exception as e:
            print(f"  Error writing output file: {e}")

    return aggregated_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate result.json files.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing i0, i1 subdirectories.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file path.")
    
    args = parser.parse_args()
    aggregate_results(args.dir, args.out)
