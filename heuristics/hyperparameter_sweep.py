#!/usr/bin/env python3
import argparse
import csv
import math
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


TARGET_PARAMS = [
    "ENERGY_MAINTENANCE_COST",
    "ENERGY_MAINTENANCE_PER_WEIGHT",
    "ENERGY_DIFFUSION_ALPHA",
    "ENERGY_FLOW_GAIN",
    "ENERGY_PULSE_LOW_RATIO",
    "ENERGY_PULSE_SINK_VALUE",
    "ENERGY_IMPORTANCE_SCALE",
    "ENERGY_DEATH_PATIENCE",
    "ENERGY_COST_EDGE_THICKEN",
    "ENERGY_COST_NEW_CONNECTION",
    "ENERGY_COST_SPROUT",
    "ENERGY_CHILD_INITIAL",
    "APOPTOSIS_WARMUP_STEPS",
    "NN_APOPTOSIS_ENERGY_GATE",
]

LINE_PATTERN = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*([^#\n\r]+)")


@dataclass
class ParamDef:
    name: str
    value: float
    is_int_style: bool


@dataclass
class RunCase:
    run_name: str
    case_type: str
    multipliers: Dict[str, float]


@dataclass
class EvalResult:
    status: str
    persistent_count: float
    persistent_ratio: float
    first_disappear_mean: float
    metrics_json: Path
    eval_csv: Path
    config_file: Path
    tuned_values: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep energy/apoptosis hyperparameters (±2%, ±3%) and evaluate "
            "persistent_after_first_connection_ratio across many runs."
        )
    )
    parser.add_argument("--base-hparams", default="hyperparameters.txt")
    parser.add_argument("--build-dir", default="cmake-build-debug")
    parser.add_argument("--mode", choices=["sweep", "hill"], default="sweep")
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--persistent-until-step", type=int, default=1075)
    parser.add_argument("--random-combos", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--deltas",
        default="0.02,0.03",
        help="Comma separated deltas used as (1±delta), e.g. 0.02,0.03",
    )
    parser.add_argument(
        "--results-root",
        default="results/hparam_sweep",
        help="Root directory for generated configs/results",
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        help="Build eval_seed_trials before sweep",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop sweep on first failed run",
    )
    parser.add_argument("--hc-iters", type=int, default=250)
    parser.add_argument("--hc-restarts", type=int, default=5)
    parser.add_argument("--hc-k", type=int, default=3, help="Parameters changed per hill step")
    parser.add_argument("--hc-k-fine", type=int, default=1, help="Parameters changed per step in fine phase")
    parser.add_argument("--hc-patience", type=int, default=40)
    parser.add_argument("--hc-min-mult", type=float, default=0.5)
    parser.add_argument("--hc-max-mult", type=float, default=1.5)
    parser.add_argument(
        "--hc-fine-deltas",
        default="0.01",
        help="Comma separated fine-phase deltas, e.g. 0.01",
    )
    parser.add_argument(
        "--hc-fine-after",
        type=float,
        default=0.6,
        help="Switch to fine phase after this fraction of iterations, in [0,1]",
    )
    parser.add_argument(
        "--hc-anneal",
        action="store_true",
        help="Enable simulated annealing acceptance for worse candidates",
    )
    parser.add_argument("--hc-temp-start", type=float, default=0.01)
    parser.add_argument("--hc-temp-end", type=float, default=0.0005)
    return parser.parse_args()


def load_hparams_text(path: Path) -> Tuple[List[str], Dict[str, ParamDef]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    params: Dict[str, ParamDef] = {}

    for line in lines:
        m = LINE_PATTERN.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        raw = m.group(2).strip()
        try:
            value = float(raw)
        except ValueError:
            continue
        is_int_style = "." not in raw and "e" not in raw.lower()
        params[key] = ParamDef(name=key, value=value, is_int_style=is_int_style)

    return lines, params


def apply_overrides(
    base_lines: List[str],
    base_defs: Dict[str, ParamDef],
    multipliers: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    tuned_values: Dict[str, float] = {}
    for key, m in multipliers.items():
        base = base_defs[key].value
        raw = base * m
        if base_defs[key].is_int_style:
            tuned = float(max(1, int(round(raw))))
        else:
            tuned = raw
        tuned_values[key] = tuned

    out_lines: List[str] = []
    for line in base_lines:
        m = LINE_PATTERN.match(line)
        if not m:
            out_lines.append(line)
            continue

        key = m.group(1).strip()
        if key not in tuned_values:
            out_lines.append(line)
            continue

        leading = line[: line.index(key)]
        trailing_comment = ""
        hash_idx = line.find("#")
        if hash_idx >= 0:
            trailing_comment = line[hash_idx:].rstrip("\n\r")

        if base_defs[key].is_int_style:
            new_value_str = str(int(round(tuned_values[key])))
        else:
            new_value_str = f"{tuned_values[key]:.10g}"

        new_line = f"{leading}{key} = {new_value_str}"
        if trailing_comment:
            new_line += f" {trailing_comment}"
        if line.endswith("\r\n"):
            new_line += "\r\n"
        elif line.endswith("\n"):
            new_line += "\n"
        out_lines.append(new_line)

    return "".join(out_lines), tuned_values


def build_cases(target_keys: List[str], deltas: List[float], random_combos: int, random_seed: int) -> List[RunCase]:
    cases: List[RunCase] = []

    cases.append(RunCase(run_name="baseline", case_type="baseline", multipliers={}))

    one_factor_mults: List[float] = []
    for d in deltas:
        one_factor_mults.extend([1.0 - d, 1.0 + d])

    for key in target_keys:
        for mult in one_factor_mults:
            suffix = f"{mult:.3f}".replace(".", "p")
            run_name = f"one_{key.lower()}_{suffix}"
            cases.append(
                RunCase(
                    run_name=run_name,
                    case_type="one_factor",
                    multipliers={key: mult},
                )
            )

    rng = random.Random(random_seed)
    random_choices = [1.0]
    for d in deltas:
        random_choices.extend([1.0 - d, 1.0 + d])

    for i in range(random_combos):
        multipliers = {key: rng.choice(random_choices) for key in target_keys}
        run_name = f"combo_{i:04d}"
        cases.append(
            RunCase(
                run_name=run_name,
                case_type="combo",
                multipliers=multipliers,
            )
        )

    return cases


def run_command(cmd: List[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def parse_deltas(deltas_text: str) -> List[float]:
    deltas = [float(x.strip()) for x in deltas_text.split(",") if x.strip()]
    if not deltas:
        raise ValueError("No valid deltas parsed from --deltas")
    for d in deltas:
        if d <= 0.0 or d >= 1.0:
            raise ValueError(f"Invalid delta {d}. Use values in (0, 1), e.g. 0.02")
    return deltas


def is_better(score_a: Tuple[float, float], score_b: Tuple[float, float]) -> bool:
    if score_a[0] != score_b[0]:
        return score_a[0] > score_b[0]
    return score_a[1] > score_b[1]


def scalar_score(score: Tuple[float, float]) -> float:
    return score[0] + (score[1] * 1.0e-6)


def anneal_accept(
    current_score: Tuple[float, float],
    candidate_score: Tuple[float, float],
    temperature: float,
    rng: random.Random,
) -> bool:
    if is_better(candidate_score, current_score):
        return True
    if temperature <= 0.0:
        return False
    delta = scalar_score(candidate_score) - scalar_score(current_score)
    prob = math.exp(delta / temperature)
    return rng.random() < prob


def get_start_run_index(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    max_idx = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_idx = max(max_idx, int(row.get("run_index", 0)))
            except (TypeError, ValueError):
                continue
    return max_idx


def evaluate_case(
    repo_root: Path,
    exe_path: Path,
    analyze_script: Path,
    build_dir: Path,
    run_dir: Path,
    run_index: int,
    run_name: str,
    multipliers: Dict[str, float],
    base_lines: List[str],
    param_defs: Dict[str, ParamDef],
    args: argparse.Namespace,
) -> EvalResult:
    config_text, tuned_values = apply_overrides(base_lines, param_defs, multipliers)
    config_path = run_dir / "hyperparameters.txt"
    config_path.write_text(config_text, encoding="utf-8")

    eval_csv = build_dir / f"seed_step_metrics_{run_index:05d}.csv"
    metrics_dir = run_dir / "connected_metrics"
    metrics_json = metrics_dir / "metrics_overview.json"

    run_command(
        [
            str(exe_path),
            str(config_path),
            str(eval_csv),
            str(args.trials),
            str(args.steps),
            str(args.cols),
            str(args.rows),
            str(args.seed_start),
            "0",
            str(args.threads),
        ],
        repo_root,
    )

    run_command(
        [
            "python",
            str(analyze_script),
            str(eval_csv),
            "--out-dir",
            str(metrics_dir),
            "--persistent-until-step",
            str(args.persistent_until_step),
        ],
        repo_root,
    )

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    first_disappear_stats = metrics.get("first_disappear_step_stats", {})

    return EvalResult(
        status="ok",
        persistent_count=float(metrics.get("persistent_after_first_connection_count", 0.0)),
        persistent_ratio=float(metrics.get("persistent_after_first_connection_ratio", 0.0)),
        first_disappear_mean=float(first_disappear_stats.get("mean", 0.0) or 0.0),
        metrics_json=metrics_json,
        eval_csv=eval_csv,
        config_file=config_path,
        tuned_values=tuned_values,
    )


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base_hparams = (repo_root / args.base_hparams).resolve()
    build_dir = (repo_root / args.build_dir).resolve()
    exe_path = build_dir / "eval_seed_trials.exe"
    analyze_script = repo_root / "results" / "analyze_connected_metrics.py"

    results_root = (repo_root / args.results_root).resolve()
    run_root = results_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    if not base_hparams.exists():
        raise FileNotFoundError(f"Base hyperparameter file not found: {base_hparams}")
    if not analyze_script.exists():
        raise FileNotFoundError(f"Analysis script not found: {analyze_script}")

    if args.build_first:
        run_command(["cmake", "--build", str(build_dir), "--target", "eval_seed_trials", "-j"], repo_root)

    if not exe_path.exists():
        raise FileNotFoundError(f"Evaluator executable not found: {exe_path}")

    base_lines, param_defs = load_hparams_text(base_hparams)
    missing_keys = [k for k in TARGET_PARAMS if k not in param_defs]
    if missing_keys:
        raise ValueError(f"Missing target parameters in base hyperparameters: {missing_keys}")

    deltas = parse_deltas(args.deltas)
    csv_path = results_root / ("hill_results.csv" if args.mode == "hill" else "sweep_results.csv")
    csv_exists = csv_path.exists()
    fieldnames = [
        "run_index",
        "run_name",
        "case_type",
        "mode",
        "restart",
        "iteration",
        "accepted",
        "status",
        "persistent_after_first_connection_count",
        "persistent_after_first_connection_ratio",
        "first_disappear_step_mean",
        "metrics_json",
        "eval_csv",
        "config_file",
    ] + TARGET_PARAMS

    with csv_path.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        run_index = get_start_run_index(csv_path)
        if run_index > 0:
            print(f"Resuming: next run_index starts at {run_index + 1}")

        if args.mode == "sweep":
            cases = build_cases(TARGET_PARAMS, deltas, args.random_combos, args.random_seed)
            print(f"Total sweep runs planned: {len(cases)}")

            for case in cases:
                run_index += 1
                run_dir = run_root / f"{run_index:05d}_{case.run_name}"
                run_dir.mkdir(parents=True, exist_ok=True)

                row = {
                    "run_index": run_index,
                    "run_name": case.run_name,
                    "case_type": case.case_type,
                    "mode": "sweep",
                    "restart": 0,
                    "iteration": 0,
                    "accepted": "",
                    "status": "ok",
                    "persistent_after_first_connection_count": "",
                    "persistent_after_first_connection_ratio": "",
                    "first_disappear_step_mean": "",
                    "metrics_json": "",
                    "eval_csv": "",
                    "config_file": "",
                }

                try:
                    result = evaluate_case(
                        repo_root,
                        exe_path,
                        analyze_script,
                        build_dir,
                        run_dir,
                        run_index,
                        case.run_name,
                        case.multipliers,
                        base_lines,
                        param_defs,
                        args,
                    )

                    row["persistent_after_first_connection_count"] = result.persistent_count
                    row["persistent_after_first_connection_ratio"] = result.persistent_ratio
                    row["first_disappear_step_mean"] = result.first_disappear_mean
                    row["metrics_json"] = str(result.metrics_json)
                    row["eval_csv"] = str(result.eval_csv)
                    row["config_file"] = str(result.config_file)

                    for key in TARGET_PARAMS:
                        row[key] = result.tuned_values.get(key, param_defs[key].value)

                except Exception as exc:
                    row["status"] = f"error: {exc}"
                    print(f"[ERROR] run {run_index} ({case.run_name}) -> {exc}")
                    writer.writerow(row)
                    f_csv.flush()
                    if args.stop_on_error:
                        raise
                    continue

                writer.writerow(row)
                f_csv.flush()
                print(
                    f"[OK] {run_index}/{len(cases)} {case.run_name} "
                    f"ratio={row['persistent_after_first_connection_ratio']}"
                )

        else:
            rng = random.Random(args.random_seed)
            local_choices = [1.0]
            for d in deltas:
                local_choices.extend([1.0 - d, 1.0 + d])
            fine_deltas = parse_deltas(args.hc_fine_deltas)
            fine_after = min(1.0, max(0.0, args.hc_fine_after))

            best_global_score = (-1.0, -1.0)
            best_global_name = ""

            print(
                f"Hill-climb mode: restarts={args.hc_restarts}, "
                f"iters={args.hc_iters}, k={args.hc_k}, patience={args.hc_patience}"
            )

            for restart in range(args.hc_restarts):
                current = {k: 1.0 for k in TARGET_PARAMS}
                if restart > 0:
                    for key in TARGET_PARAMS:
                        current[key] *= rng.choice(local_choices)
                        current[key] = max(args.hc_min_mult, min(args.hc_max_mult, current[key]))

                run_index += 1
                init_name = f"hc_r{restart:02d}_init"
                run_dir = run_root / f"{run_index:05d}_{init_name}"
                run_dir.mkdir(parents=True, exist_ok=True)

                init_row = {
                    "run_index": run_index,
                    "run_name": init_name,
                    "case_type": "hill_init",
                    "mode": "hill",
                    "restart": restart,
                    "iteration": 0,
                    "accepted": 1,
                    "status": "ok",
                    "persistent_after_first_connection_count": "",
                    "persistent_after_first_connection_ratio": "",
                    "first_disappear_step_mean": "",
                    "metrics_json": "",
                    "eval_csv": "",
                    "config_file": "",
                }

                try:
                    init_result = evaluate_case(
                        repo_root,
                        exe_path,
                        analyze_script,
                        build_dir,
                        run_dir,
                        run_index,
                        init_name,
                        current,
                        base_lines,
                        param_defs,
                        args,
                    )
                    init_score = (init_result.persistent_ratio, init_result.persistent_count)
                    for key in TARGET_PARAMS:
                        init_row[key] = init_result.tuned_values.get(key, param_defs[key].value)
                    init_row["persistent_after_first_connection_count"] = init_result.persistent_count
                    init_row["persistent_after_first_connection_ratio"] = init_result.persistent_ratio
                    init_row["first_disappear_step_mean"] = init_result.first_disappear_mean
                    init_row["metrics_json"] = str(init_result.metrics_json)
                    init_row["eval_csv"] = str(init_result.eval_csv)
                    init_row["config_file"] = str(init_result.config_file)
                except Exception as exc:
                    init_row["status"] = f"error: {exc}"
                    writer.writerow(init_row)
                    f_csv.flush()
                    print(f"[ERROR] restart {restart} init -> {exc}")
                    if args.stop_on_error:
                        raise
                    continue

                writer.writerow(init_row)
                f_csv.flush()
                print(f"[OK] {init_name} ratio={init_row['persistent_after_first_connection_ratio']}")

                current_score = init_score
                no_improve = 0

                if is_better(current_score, best_global_score):
                    best_global_score = current_score
                    best_global_name = init_name

                for iteration in range(1, args.hc_iters + 1):
                    candidate = dict(current)
                    is_fine_phase = (iteration / max(1, args.hc_iters)) >= fine_after
                    active_deltas = fine_deltas if is_fine_phase else deltas
                    active_k = args.hc_k_fine if is_fine_phase else args.hc_k
                    k = max(1, min(active_k, len(TARGET_PARAMS)))
                    changed = rng.sample(TARGET_PARAMS, k)
                    for key in changed:
                        delta = rng.choice(active_deltas)
                        direction = rng.choice([-1.0, 1.0])
                        candidate[key] *= (1.0 + direction * delta)
                        candidate[key] = max(args.hc_min_mult, min(args.hc_max_mult, candidate[key]))

                    if args.hc_iters <= 1:
                        progress = 1.0
                    else:
                        progress = (iteration - 1) / (args.hc_iters - 1)
                    temperature = args.hc_temp_start + (args.hc_temp_end - args.hc_temp_start) * progress

                    run_index += 1
                    run_name = f"hc_r{restart:02d}_it{iteration:04d}"
                    run_dir = run_root / f"{run_index:05d}_{run_name}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    row = {
                        "run_index": run_index,
                        "run_name": run_name,
                        "case_type": "hill_step",
                        "mode": "hill",
                        "restart": restart,
                        "iteration": iteration,
                        "accepted": 0,
                        "status": "ok",
                        "persistent_after_first_connection_count": "",
                        "persistent_after_first_connection_ratio": "",
                        "first_disappear_step_mean": "",
                        "metrics_json": "",
                        "eval_csv": "",
                        "config_file": "",
                    }

                    try:
                        result = evaluate_case(
                            repo_root,
                            exe_path,
                            analyze_script,
                            build_dir,
                            run_dir,
                            run_index,
                            run_name,
                            candidate,
                            base_lines,
                            param_defs,
                            args,
                        )
                        score = (result.persistent_ratio, result.persistent_count)
                        if args.hc_anneal:
                            accepted = anneal_accept(current_score, score, temperature, rng)
                        else:
                            accepted = is_better(score, current_score)
                        if accepted:
                            current = candidate
                            current_score = score
                            no_improve = 0
                            row["accepted"] = 1
                        else:
                            no_improve += 1

                        if is_better(score, best_global_score):
                            best_global_score = score
                            best_global_name = run_name

                        row["persistent_after_first_connection_count"] = result.persistent_count
                        row["persistent_after_first_connection_ratio"] = result.persistent_ratio
                        row["first_disappear_step_mean"] = result.first_disappear_mean
                        row["metrics_json"] = str(result.metrics_json)
                        row["eval_csv"] = str(result.eval_csv)
                        row["config_file"] = str(result.config_file)
                        for key in TARGET_PARAMS:
                            row[key] = result.tuned_values.get(key, param_defs[key].value)

                    except Exception as exc:
                        row["status"] = f"error: {exc}"
                        print(f"[ERROR] {run_name} -> {exc}")
                        if args.stop_on_error:
                            writer.writerow(row)
                            f_csv.flush()
                            raise

                    writer.writerow(row)
                    f_csv.flush()
                    print(
                        f"[HC] restart={restart} iter={iteration} "
                        f"ratio={row['persistent_after_first_connection_ratio']} "
                        f"accepted={row['accepted']} best={best_global_score[0]} temp={temperature:.6f}"
                    )

                    if no_improve >= args.hc_patience:
                        print(f"[HC] restart={restart} early-stop at iter={iteration} (patience)")
                        break

            print(
                f"[HC] best ratio={best_global_score[0]} count={best_global_score[1]} "
                f"run={best_global_name}"
            )

    print(f"\n{args.mode.capitalize()} completed.")
    print(f"- Results CSV: {csv_path}")
    print(f"- Per-run artifacts: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
