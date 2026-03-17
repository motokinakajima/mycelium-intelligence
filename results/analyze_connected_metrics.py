#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze seed-step connectivity metrics and export disappearance/persistence reports."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="cmake-build-debug/seed_step_metrics_500.csv",
        help="Path to input CSV (default: cmake-build-debug/seed_step_metrics_500.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/connected_metrics",
        help="Directory for output reports (default: results/connected_metrics)",
    )
    parser.add_argument(
        "--persistent-until-step",
        type=int,
        default=1075,
        help="Step threshold for persistence metric (default: 1075)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    seed_steps = defaultdict(list)
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"seed", "step", "connected", "node_count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            seed = int(row["seed"])
            step = int(row["step"])
            connected = int(row["connected"])
            node_count = int(row["node_count"])
            seed_steps[seed].append((step, connected, node_count))

    disappear_events = []
    seed_summary = []

    persistent_until_step = max(0, args.persistent_until_step)

    for seed in sorted(seed_steps.keys()):
        records = sorted(seed_steps[seed], key=lambda x: x[0])

        prev_connected = None
        drop_steps = []
        first_connected_step = None
        last_connected_step = None
        final_connected = records[-1][1]
        final_node_count = records[-1][2]

        connected_steps = 0
        for step, connected, _node_count in records:
            if connected == 1:
                connected_steps += 1
                if first_connected_step is None:
                    first_connected_step = step
                last_connected_step = step

            if prev_connected == 1 and connected == 0:
                drop_steps.append(step)
                disappear_events.append({"seed": seed, "step": step})

            prev_connected = connected

        ever_connected = first_connected_step is not None
        max_step = records[-1][0]
        effective_threshold = min(persistent_until_step, max_step)
        connected_at_threshold = 0
        for step, connected, _node_count in records:
            if step == effective_threshold:
                connected_at_threshold = connected
                break

        drop_steps_until_threshold = [s for s in drop_steps if s <= effective_threshold]
        persistent_after_first_connection = (
            ever_connected
            and first_connected_step <= effective_threshold
            and len(drop_steps_until_threshold) == 0
            and connected_at_threshold == 1
        )
        always_connected = connected_steps == len(records)

        seed_summary.append(
            {
                "seed": seed,
                "num_steps": len(records),
                "connected_steps": connected_steps,
                "connected_ratio": connected_steps / len(records),
                "ever_connected": int(ever_connected),
                "first_connected_step": first_connected_step if ever_connected else "",
                "last_connected_step": last_connected_step if ever_connected else "",
                "num_disappear_events": len(drop_steps),
                "first_disappear_step": drop_steps[0] if drop_steps else "",
                "final_connected": final_connected,
                "final_node_count": final_node_count,
                "persistent_after_first_connection": int(persistent_after_first_connection),
                "persistent_threshold_step": effective_threshold,
                "always_connected": int(always_connected),
            }
        )

    persistent_seeds = [
        row
        for row in seed_summary
        if row["persistent_after_first_connection"] == 1
    ]

    disappear_csv = out_dir / "connected_disappear_events.csv"
    summary_csv = out_dir / "seed_connectivity_summary.csv"
    persistent_csv = out_dir / "persistent_seeds_metrics.csv"
    metrics_json = out_dir / "metrics_overview.json"

    with disappear_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "step"])
        writer.writeheader()
        writer.writerows(disappear_events)

    summary_fields = [
        "seed",
        "num_steps",
        "connected_steps",
        "connected_ratio",
        "ever_connected",
        "first_connected_step",
        "last_connected_step",
        "num_disappear_events",
        "first_disappear_step",
        "final_connected",
        "final_node_count",
        "persistent_after_first_connection",
        "persistent_threshold_step",
        "always_connected",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(seed_summary)

    with persistent_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(persistent_seeds)

    total_seeds = len(seed_summary)
    first_disappear_steps = [
        int(r["first_disappear_step"])
        for r in seed_summary
        if r["first_disappear_step"] != ""
    ]

    if first_disappear_steps:
        first_disappear_stats = {
            "count": len(first_disappear_steps),
            "mean": statistics.fmean(first_disappear_steps),
            "median": statistics.median(first_disappear_steps),
            "min": min(first_disappear_steps),
            "max": max(first_disappear_steps),
        }
    else:
        first_disappear_stats = {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    metrics = {
        "input_csv": str(input_path),
        "total_seeds": total_seeds,
        "seeds_ever_connected": sum(r["ever_connected"] for r in seed_summary),
        "seeds_with_disappear_events": sum(1 for r in seed_summary if r["num_disappear_events"] > 0),
        "total_disappear_events": len(disappear_events),
        "first_disappear_step_stats": first_disappear_stats,
        "persistent_threshold_step": persistent_until_step,
        "persistent_after_first_connection_count": len(persistent_seeds),
        "persistent_after_first_connection_ratio": (
            len(persistent_seeds) / total_seeds if total_seeds else 0.0
        ),
        "always_connected_count": sum(r["always_connected"] for r in seed_summary),
        "final_connected_count": sum(r["final_connected"] for r in seed_summary),
        "outputs": {
            "connected_disappear_events": str(disappear_csv),
            "seed_connectivity_summary": str(summary_csv),
            "persistent_seeds_metrics": str(persistent_csv),
            "metrics_overview": str(metrics_json),
        },
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Analysis complete.")
    print(f"- Input: {input_path}")
    print(f"- Seeds: {total_seeds}")
    print(f"- Disappear events: {len(disappear_events)}")
    print(f"- First disappear step mean: {first_disappear_stats['mean']}")
    print(f"- Persistent threshold step: {persistent_until_step}")
    print(f"- Persistent seeds: {len(persistent_seeds)}")
    print(f"- Output dir: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
