import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# --- 1. 生成模擬數據 ---
# 狀態 A: 正常壓力
normal_state = np.random.normal(loc=100, scale=2, size=200)

# 狀態 B: 壓力轉移後
shifted_state = np.random.normal(loc=120, scale=2, size=200)

# 加上一個短暫的過渡期
transition = np.linspace(102, 118, 10)

# 合併成完整的時間序列
time_series = np.concatenate([normal_state, transition, shifted_state])

# --- 2. 滑動窗口特徵工程 ---
window_size = 5
X = []
# 從時間序列中提取滑動窗口作為特徵向量
for i in range(len(time_series) - window_size):
    X.append(time_series[i:i+window_size])

X = np.array(X)

# --- 3. 應用 DBSCAN ---
# eps: 兩個樣本被視為鄰居的最大距離
# min_samples: 一個點被視為核心點所需的鄰域樣本數
dbscan = DBSCAN(eps=5, min_samples=5)
labels = dbscan.fit_predict(X)

# 由於標籤數量比原始數據少 (因為窗口)，我們需要對齊
# 簡單起見，我們將標籤填充到與原始數據等長，以便繪圖
aligned_labels = np.full(len(time_series), -2) # 用-2代表未標記
aligned_labels[window_size:] = labels

# --- 4. 視覺化結果 ---
plt.figure(figsize=(15, 6))

# 根據 DBSCAN 的標籤來繪製不同顏色的點
plt.scatter(np.where(aligned_labels == 0)[0], time_series[aligned_labels == 0], color='blue', label='狀態 A (Cluster 0)', s=10)
plt.scatter(np.where(aligned_labels == 1)[0], time_series[aligned_labels == 1], color='green', label='狀態 B (Cluster 1)', s=10)
plt.scatter(np.where(aligned_labels == -1)[0], time_series[aligned_labels == -1], color='red', label='轉變點 (Noise)', s=50, marker='x')

plt.title("使用 DBSCAN 偵測時間序列轉變")
plt.xlabel("時間點")
plt.ylabel("感測器壓力 (Pa)")
plt.legend()
plt.grid(True)
plt.show()

print(f"找到的群集標籤: {np.unique(labels)}")