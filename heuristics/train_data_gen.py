import numpy as np
import csv

NUM_SAMPLES = 25000
OUTPUT_FILE = 'training_data_v11_safe_junctions.csv' # ← バッチ変更用

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def generate_sample():
    pressure = np.random.uniform(-1.0, 1.0, 2) 
    target = np.random.uniform(-1.0, 1.0, 2)
    target = normalize(target)
    
    flow_com_raw = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)])
    flow_mag = min(1.0, np.linalg.norm(flow_com_raw)) 
    flow_dir = normalize(flow_com_raw)

    importance = np.random.uniform(0.0, 1.0)
    crowdedness = np.random.uniform(0.0, 1.0)

    # ---------------------------------------------------------
    # 【修正版】交差点での暴走を防ぐリミッター付きモデル
    # ---------------------------------------------------------
    
    starvation = 1.0 - importance # 飢餓度（最大1.0）
    stagnation = 1.0 - flow_mag   # 淀み度（最大1.0）

    # 1. Shear Stress (せん断応力) はそのまま維持！
    # 命綱（橋）を守るための重要な指標。
    shear_stress = flow_mag / (importance + 0.1) 
    
    # 2. Pruneの計算（ここが暴走の原因でした！）
    # 割り算を使うのをやめ、「淀み」と「密集度」の掛け算にします（最大でも1.0にしかならない）
    desired_prune = stagnation * crowdedness

    # 【絶対的なリミッター】
    # どれだけPruneしたくても、「自分が飢えている度合い(starvation)」を上限とする！
    # 交差点(importanceが高い＝starvationが低い)なら、ここで強制的にPruneが極小に抑えられます。
    base_prune = min(desired_prune, starvation)
    
    # 命綱(shear_stressが高い)なら、さらにPruneを減らす
    prune_strength = max(0.0, base_prune - (shear_stress * 0.3))

    # --- Grow (強化) ---
    grow_dir = flow_dir * 1.0 + target * 0.2
    # 命綱として負担がかかっている（shear_stressが高い）、または普通に流れているならGrow
    grow_strength = min(1.0, max(flow_mag, shear_stress * 0.4))

    # --- 方向の計算 ---
    prune_dir = -flow_dir * 1.0 + pressure * 1.0
    grow = normalize(grow_dir) * grow_strength if np.linalg.norm(grow_dir) > 0 else np.array([0.0, 0.0])
    prune = normalize(prune_dir) * prune_strength if np.linalg.norm(prune_dir) > 0 else np.array([0.0, 0.0])

    shift_dir = target * 0.5 + flow_dir * 0.5 - pressure * 1.0
    shift = normalize(shift_dir) * 0.2 if np.linalg.norm(shift_dir) > 0 else np.array([0.0, 0.0])

    apoptosis = -1.0

    inputs = [
        pressure[0], pressure[1],
        target[0], target[1],
        flow_com_raw[0], flow_com_raw[1],
        importance, crowdedness
    ]

    ideal_outputs = [
        grow[0], grow[1],
        prune[0], prune[1],
        shift[0], shift[1],
        apoptosis
    ]

    final_outputs = []
    for val in ideal_outputs:
        noisy_val = val + np.random.normal(0, 0.03)
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