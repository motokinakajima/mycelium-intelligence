import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_NUM_SAMPLES = 25000
DEFAULT_SEED = 42
OUTPUT_FILE = Path(__file__).with_name("training_data.csv")


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1.0e-8:
        return np.zeros_like(v)
    return v / norm


def clamp_tanh_range(values):
    return np.clip(values, -1.0, 1.0)


def generate_sample(rng):
    # Inputs (8): pressure(2), target(2), flow_com(2), importance, crowdedness
    pressure = rng.uniform(-1.0, 1.0, 2)

    target = rng.uniform(-1.0, 1.0, 2)
    target = normalize(target)

    flow_dir = normalize(rng.uniform(-1.0, 1.0, 2))
    flow_mag = rng.uniform(0.0, 2.5)
    flow_com = flow_dir * flow_mag

    # importance は energy 相当を想定（0~5程度）
    importance = rng.uniform(0.0, 5.0)
    # crowdedness は近傍ノード数相当（0~8程度）
    crowdedness = rng.uniform(0.0, 8.0)

    # 入力側は C++ 側で最終 clamp される設計に合わせ、ここでも軽く制限
    importance_in = float(np.clip(importance, 0.0, 5.0))
    crowdedness_in = float(np.clip(crowdedness, 0.0, 8.0))

    inputs = np.array([
        pressure[0], pressure[1],
        target[0], target[1],
        flow_com[0], flow_com[1],
        importance_in, crowdedness_in
    ], dtype=np.float32)

    # Heuristic labels
    # Grow: target 方向を主軸、壁反発とフローも加味
    grow_vec = 1.3 * target - 0.9 * pressure + 0.25 * normalize(flow_com)
    grow = normalize(grow_vec)

    # Prune: target 逆向き + 混雑時に強める
    prune_gain = np.clip(crowdedness_in / 8.0, 0.0, 1.0)
    prune_vec = normalize(-target + 0.2 * normalize(flow_com)) * prune_gain

    # Shift: 壁から離れつつ、局所フロー/ゴールへ整流
    shift_vec = -1.0 * pressure + 0.7 * normalize(flow_com) + 0.3 * target
    shift = normalize(shift_vec)

    # Apoptosis: 低energy & 高crowdednessで上昇
    # importance(energy) が高いと生存側（負）へ、crowdedness が高いと死側（正）へ
    apoptosis_logit = 0.45 * crowdedness_in - 0.85 * importance_in - 0.20
    apoptosis = float(np.tanh(apoptosis_logit))

    ideal_outputs = np.array([
        grow[0], grow[1],
        prune_vec[0], prune_vec[1],
        shift[0], shift[1],
        apoptosis
    ], dtype=np.float32)

    noise = rng.normal(0.0, 0.12, size=7).astype(np.float32)
    final_outputs = clamp_tanh_range(ideal_outputs + noise)

    return np.concatenate([inputs, final_outputs])


def main():
    parser = argparse.ArgumentParser(description="Generate heuristic training_data.csv for node_nn")
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    num_samples = max(1, args.samples)
    rng = np.random.default_rng(args.seed)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "# in_press_x", "in_press_y", "in_target_x", "in_target_y",
            "in_flow_x", "in_flow_y", "in_importance", "in_crowdedness",
            "out_grow_x", "out_grow_y", "out_prune_x", "out_prune_y",
            "out_shift_x", "out_shift_y", "out_apoptosis"
        ]
        writer.writerow(header)

        for _ in range(num_samples):
            row = generate_sample(rng)
            writer.writerow([round(float(v), 6) for v in row])

    print(f"✅ {num_samples}件のトレーニングデータを {OUTPUT_FILE} に生成しました")


if __name__ == "__main__":
    main()