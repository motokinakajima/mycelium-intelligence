#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot persistent ratio decay from connected_disappear_events.csv "
            "using x = step / 15."
        )
    )
    parser.add_argument(
        "--events-csv",
        default="results/connected_metrics/connected_disappear_events.csv",
        help="Path to connected_disappear_events.csv",
    )
    parser.add_argument(
        "--metrics-json",
        default="results/connected_metrics/metrics_overview.json",
        help="Path to metrics_overview.json (used for total seeds fallback)",
    )
    parser.add_argument(
        "--total-seeds",
        type=int,
        default=None,
        help="Total seeds. If omitted, read from metrics_overview.json.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Max raw step for plotting. Default: max first disappear step.",
    )
    parser.add_argument(
        "--step-divisor",
        type=float,
        default=15.0,
        help="x-axis divisor (default: 15.0)",
    )
    parser.add_argument(
        "--ref-x",
        type=float,
        default=43.0,
        help="Reference x for vertical line (default: 43)",
    )
    parser.add_argument(
        "--ref-percent",
        type=float,
        default=83.6,
        help="Reference percent annotation at ref-x (default: 83.6)",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=60.0,
        help="Max x-axis value to display (default: 60.0)",
    )
    parser.add_argument(
        "--out-png",
        default="results/connected_metrics/persistent_decay_plot.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window before saving",
    )
    parser.add_argument(
        "--out-series-csv",
        default="results/connected_metrics/persistent_decay_series.csv",
        help="Output series CSV path",
    )
    return parser.parse_args()


def load_total_seeds(total_seeds_arg: int | None, metrics_json_path: Path) -> int:
    if total_seeds_arg is not None:
        if total_seeds_arg <= 0:
            raise ValueError("--total-seeds must be positive")
        return total_seeds_arg

    if metrics_json_path.exists():
        metrics = json.loads(metrics_json_path.read_text(encoding="utf-8"))
        total = int(metrics.get("total_seeds", 0))
        if total > 0:
            return total

    raise ValueError(
        "Could not determine total seeds. Provide --total-seeds or ensure metrics_overview.json has total_seeds."
    )


def main() -> int:
    args = parse_args()

    events_csv = Path(args.events_csv)
    metrics_json = Path(args.metrics_json)
    out_png = Path(args.out_png)
    out_series_csv = Path(args.out_series_csv)

    if not events_csv.exists():
        raise FileNotFoundError(f"Events CSV not found: {events_csv}")

    total_seeds = load_total_seeds(args.total_seeds, metrics_json)

    first_disappear_step_by_seed: dict[int, int] = {}
    with events_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"seed", "step"}
        if set(reader.fieldnames or []) != required:
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"Invalid columns in events CSV. Missing: {sorted(missing)}")

        for row in reader:
            seed = int(row["seed"])
            step = int(row["step"])
            prev = first_disappear_step_by_seed.get(seed)
            if prev is None or step < prev:
                first_disappear_step_by_seed[seed] = step

    if args.max_step is not None:
        max_step = max(0, args.max_step)
    else:
        max_step = max(first_disappear_step_by_seed.values(), default=0)

    first_steps = sorted(first_disappear_step_by_seed.values())
    dropped_so_far = 0
    idx = 0

    xs: list[float] = []
    ratios: list[float] = []
    counts: list[int] = []

    for step in range(max_step + 1):
        while idx < len(first_steps) and first_steps[idx] <= step:
            dropped_so_far += 1
            idx += 1

        persistent_count = total_seeds - dropped_so_far
        persistent_ratio = persistent_count / total_seeds

        xs.append(step / args.step_divisor)
        ratios.append(persistent_ratio)
        counts.append(persistent_count)

    out_series_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_series_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "step_divided", "persistent_count", "persistent_ratio"])
        for step in range(max_step + 1):
            writer.writerow([step, xs[step], counts[step], ratios[step]])

    ref_step = int(round(args.ref_x * args.step_divisor))
    if ref_step < 0:
        ref_step = 0
    if ref_step > max_step:
        ref_step = max_step
    ref_actual_ratio = ratios[ref_step]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xs, ratios, linewidth=2.0, label="Persistent ratio")
    ax.axvline(args.ref_x, linestyle="--", linewidth=1.5, label=f"x={args.ref_x:g}")
    ax.axhline(args.ref_percent / 100.0, linestyle=":", linewidth=1.2, label=f"{args.ref_percent:.1f}%")
    ax.scatter([args.ref_x], [args.ref_percent / 100.0], zorder=3)
    ax.annotate(
        f"x={args.ref_x:g}, {args.ref_percent:.1f}%",
        xy=(args.ref_x, args.ref_percent / 100.0),
        xytext=(args.ref_x + 1.5, min(0.98, args.ref_percent / 100.0 + 0.05)),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax.set_title("Persistent Ratio Decay")
    ax.set_xlabel("step")
    ax.set_ylabel("persistent ratio")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, max(0.0, args.x_max))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_series_csv_abs = out_series_csv.resolve()
    out_png_abs = out_png.resolve()

    print(f"- Series CSV (saved): {out_series_csv_abs}")
    print(f"- Plot PNG (will save): {out_png_abs}")
    if not args.no_show:
        print("- Showing plot window (close it to continue saving)...")
        plt.show()

    fig.savefig(out_png_abs, dpi=150)
    plt.close(fig)

    print("Persistent decay visualization complete.")
    print(f"- Total seeds: {total_seeds}")
    print(f"- Max step: {max_step}")
    print(f"- Ref step (x*15): {ref_step}")
    print(f"- Actual ratio at ref step: {ref_actual_ratio * 100.0:.3f}%")
    print(f"- Annotated ratio: {args.ref_percent:.1f}%")
    print(f"- Series CSV: {out_series_csv_abs}")
    print(f"- Plot PNG: {out_png_abs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
