import numpy as np
import csv

# 設定
NUM_SAMPLES = 25000
OUTPUT_FILE = 'training_data_zero_vibe.csv'

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def generate_sample():
    # ---------------------------------------------------------
    # 1. 入力値のランダム生成 (Inputs: 8 dimensions)
    # ---------------------------------------------------------
    pressure = np.random.uniform(-1.0, 1.0, 2) 
    target = np.random.uniform(-1.0, 1.0, 2)
    target = normalize(target)
    flow_com = np.random.uniform(-1.0, 1.0, 2)
    importance = np.random.uniform(0.0, 1.0)
    crowdedness = np.random.uniform(0.0, 1.0)

    inputs = [
        pressure[0], pressure[1],
        target[0], target[1],
        flow_com[0], flow_com[1],
        importance, crowdedness
    ]

    # ---------------------------------------------------------
    # 2. 「完全な無の境地（Zero Vibe）」の出力
    # ---------------------------------------------------------
    # Grow (Output 0, 1): 成長しない (0.0, 0.0)
    # Prune (Output 2, 3): 刈り込まない (0.0, 0.0)
    # Shift (Output 4, 5): 動かない (0.0, 0.0)
    # Apoptosis (Output 6): 自殺しない（-1.0 にして物理エンジンによる餓死に100%委ねる）
    
    ideal_outputs = [
        0.0, 0.0,  # Grow
        0.0, 0.0,  # Prune
        0.0, 0.0,  # Shift
        -1.0       # Apoptosis
    ]

    # ---------------------------------------------------------
    # 3. ノイズ付与（極小）
    # ---------------------------------------------------------
    final_outputs = []
    for val in ideal_outputs:
        # NNの学習が「完全に同一の値」で勾配消失・崩壊しないよう、
        # 影響が出ないレベルの極小の揺らぎ（標準偏差0.01）だけ付与します
        noisy_val = val + np.random.normal(0, 0.01)
        clamped_val = np.clip(noisy_val, -1.0, 1.0)
        final_outputs.append(clamped_val)

    return inputs + final_outputs

# CSVファイルへの書き込み
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [
        "# in_press_x", "in_press_y", "in_target_x", "in_target_y",
        "in_flow_x", "in_flow_y", "in_importance", "in_crowdedness",
        "out_grow_x", "out_grow_y", "out_prune_x", "out_prune_y",
        "out_shift_x", "out_shift_y", "out_apoptosis"
    ]
    writer.writerow(header)
    
    for _ in range(NUM_SAMPLES):
        row = generate_sample()
        row = [round(val, 6) for val in row]
        writer.writerow(row)

print(f"✅ {NUM_SAMPLES}件の【Zero Vibe】データを {OUTPUT_FILE} に生成しました！")