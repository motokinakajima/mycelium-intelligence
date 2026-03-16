import numpy as np
import csv

NUM_SAMPLES = 25000
OUTPUT_FILE = 'training_data_conservative.csv'

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def generate_sample():
    # 1. 入力生成
    pressure = np.random.uniform(-1.0, 1.0, 2) 
    target = np.random.uniform(-1.0, 1.0, 2)  # 一番近いノードへの方向ベクトル
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
    # 2. 「休眠と保守的刈り込み」モデル (Heuristics v6)
    # ---------------------------------------------------------
    
    # Grow: 血流(flow_com)と仲間(target)の方向を穏やかに強化
    grow_dir = flow_com * 1.0 + target * 0.5
    grow_strength = importance * 0.5  # 以前より出力を抑えめに（最大0.5）
    grow = normalize(grow_dir) * grow_strength if np.linalg.norm(grow_dir) > 0 else np.array([0.0, 0.0])

    # Prune: 【超保守的】エネルギーがあるうちは絶対に切らない
    prune_dir = -flow_com * 1.0 - target * 0.5 - pressure * 0.5
    # importance が 0.8 を下回って初めて、ジワジワと切り始める（最大出力も0.2に制限）
    prune_strength = max(0.0, 0.8 - importance) * 0.2
    prune = normalize(prune_dir) * prune_strength if np.linalg.norm(prune_dir) > 0 else np.array([0.0, 0.0])

    # Shift: 控えめにうにょうにょさせる
    shift_dir = target * 0.5 + flow_com * 0.5 - pressure * 1.0
    shift = normalize(shift_dir) * 0.2 if np.linalg.norm(shift_dir) > 0 else np.array([0.0, 0.0])

    # Apoptosis: 物理エンジンにお任せ
    apoptosis = -1.0

    ideal_outputs = [
        grow[0], grow[1],
        prune[0], prune[1],
        shift[0], shift[1],
        apoptosis
    ]

    # 3. ノイズ付与（ノイズも小さくして暴発を防ぐ）
    final_outputs = []
    for val in ideal_outputs:
        noisy_val = val + np.random.normal(0, 0.02)
        clamped_val = np.clip(noisy_val, -1.0, 1.0)
        final_outputs.append(clamped_val)

    return inputs + final_outputs
# CSVファイルへの書き込み
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # C++コードが読み飛ばしてくれるヘッダー行を追加
    header = [
        "# in_press_x", "in_press_y", "in_target_x", "in_target_y",
        "in_flow_x", "in_flow_y", "in_importance", "in_crowdedness",
        "out_grow_x", "out_grow_y", "out_prune_x", "out_prune_y",
        "out_shift_x", "out_shift_y", "out_apoptosis"
    ]
    writer.writerow(header)
    
    # データの生成と書き込み
    for _ in range(NUM_SAMPLES):
        row = generate_sample()
        # 小数点以下6桁に丸めてファイルサイズを節約（C++のstofで問題なく読めます）
        row = [round(val, 6) for val in row]
        writer.writerow(row)

print(f"✅ {NUM_SAMPLES}件のトレーニングデータを {OUTPUT_FILE} に生成しました！")