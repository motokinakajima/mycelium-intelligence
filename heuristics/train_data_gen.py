import numpy as np
import csv

NUM_SAMPLES = 25000
OUTPUT_FILE = 'training_data_v12_murrays_law.csv'

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
    
    # 【超重要】流量のスケール感が小さくなっていると想定し、AIが感じ取りやすくする
    # ※C++側で正規化している場合はそのままでも良いですが、ここでは感度を上げます
    flow_mag = min(1.0, np.linalg.norm(flow_com_raw) * 3.0) # 感度を3倍にブースト！
    flow_dir = normalize(flow_com_raw)

    importance = np.random.uniform(0.0, 1.0)
    crowdedness = np.random.uniform(0.0, 1.0) # 今回は一旦無視して基本ルールをテストします

    # ---------------------------------------------------------
    # v12: 超シンプル・生物学的ルール（マレーの法則ベース）
    # ---------------------------------------------------------
    
    stagnation = 1.0 - flow_mag   
    starvation = 1.0 - importance 

    # --- Grow (強化) ---
    grow_dir = flow_dir * 1.0 + target * 0.2
    # 少しでも流れているなら、素直に太くする（命綱の保護）
    grow_strength = flow_mag * 0.8 

    # --- Prune (刈り込み) ---
    prune_dir = -flow_dir * 1.0 + pressure * 1.0
    # 「流れていない(stagnation)」 AND 「エネルギーもない(starvation)」時だけ刈り込む
    # 流量が少しでもある命綱は、stagnationが下がるのでPruneされにくくなる
    prune_strength = (stagnation * starvation) * 0.6

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
        noisy_val = val + np.random.normal(0, 0.02) # ノイズも少し減らして純粋な挙動を見ます
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