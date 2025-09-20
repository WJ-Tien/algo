import numpy as np
import matplotlib.pyplot as plt # noqa
import pandas as pd
from sklearn.decomposition import PCA


"""
PCA: principal component analysis
好的，這是一個非常棒的延伸問題！不過在我們開始之前，有一個超級重要的觀念需要先澄清一下，這在面試中絕對是個加分題。
重要觀念澄清：主成分的數量上限
一個常見的誤解是，我們可以隨意指定 n_components 的數量。事實上，主成分的數量有一個嚴格的上限：
主成分的數量 (n_components) <= 原始特徵的數量 (n_features)
"""

# 建立我們的 n=6 範例資料
data = {
    'math': [60, 65, 75, 80, 90, 98],
    'physics': [65, 68, 76, 85, 92, 99]
}
students = ['studentA', 'studentB', 'studentC', 
            'studentD', 'studentE', 'studentF']

df = pd.DataFrame(data, index=students)
X = df.values # convert to a numpy array 

# # ori data viz
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], s=100, c='blue', alpha=0.7)
# plt.title('math vs physics (ori distr)')
# plt.xlabel('math score')
# plt.ylabel('physics score')
# for i, name in enumerate(students):
#     plt.text(X[i, 0]+0.5, X[i, 1]+0.5, name) # 標上學生名字
# plt.grid(True)
# plt.axis('equal')
# plt.show()

print(df)
print("\n")

# Mean value
X_mean = np.mean(X, axis=0)
print(f"資料平均值 (數學, 物理): {X_mean}\n")

# center the value 
# move to (0, 0)
# 6 x 2
X_centered = X - X_mean
print("--- 步驟 1: 中心化後的資料 ---")
print(pd.DataFrame(X_centered, index=students, columns=['math_centered', 'physics_centered']))


# 計算協方差矩陣 (記得要轉置 .T)
# conventional --> samples * features
# numpy --> features * samples
# 2 x 2
# 如果你的資料有 n 個 features (特徵)，那麼協方差矩陣 (cov_mat) 
# 就會是一個 n x n 的方陣，裡面會有 n 個項目。
cov_mat = np.cov(X_centered.T)
print("\n--- 步驟 2: 協方差矩陣 ---")
print(cov_mat)


eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
# also 2 x 2 (n x n, same as cov_mat)
# column vector, each column is a vector
# PC1_1, PC2_1 ...
# PC1_2, PC2_2 ...
# PC1_3, PC2_3 ...
print("\n--- 步驟 3: 特徵值與特徵向量 ---")
print("特徵值 (Eigenvalues):")
print(eigen_values)
print("\n特徵向量 (Eigenvectors):")
print(eigen_vectors)


# 根據特徵值由大到小排序
# 2x2
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigenvectors = eigen_vectors[:, sorted_indices]

# 選擇最重要的特徵向量 (PC1)
# loading here !
pc1_vector = sorted_eigenvectors[:, 0]
print(f"\n最重要的主成分 (PC1) 方向向量: {pc1_vector}\n")

# 將中心化資料投影到 PC1 上
# 6x2 x 2x1 = 6 x 1
X_projected_numpy = X_centered.dot(pc1_vector)
print("--- 步驟 4: NumPy 投影後的新座標 (理科綜合能力分數) ---")
print(pd.DataFrame({'PC1_Numpy': X_projected_numpy}, index=students))


######################################


# 建立一個 PCA 模型，n_components=1 代表我們要降到 1 維
pca = PCA(n_components=1)

# 讓模型學習資料並轉換 (fit_transform 會自動完成中心化)
X_projected_sklearn = pca.fit_transform(X)
print("--- Scikit-learn 投影後的新座標 ---")
print(pd.DataFrame(X_projected_sklearn, index=students, columns=['PC1_Sklearn']))


# Loadings 儲存在 pca.components_ 屬性裡
loadings = pca.components_[0] # [0] 代表 PC1

print("\n--- 載荷 (Loadings) 解讀 ---")
print(f"PC1 的載荷向量: {loadings}")
print("\n這代表 PC1 的配方是:")
print(f"理科綜合能力 ≈ {loadings[0]:.3f} * (標準化數學) + {loadings[1]:.3f} * (標準化物理)")

print("\n--- Sklearn 計算的變異解釋比例 ---")
print(pca.explained_variance_ratio_)


"""
我們可以從三個層面來解讀這些數字：

1. 正負號 (Sign)：代表相對於「平均水平」的位置
零點 (Zero Point)：這個新分數的 0 點，代表的就是原始資料的平均點（平均數學成績, 平均物理成績）。
正數 (+)：表示該學生的「理科綜合能力」在平均水平之上。例如學生D、E、F。
負數 (-)：表示該學生的「理科綜合能力」在平均水平之下。例如學生A、B、C。

2. 大小 (Magnitude)：代表偏離「平均水平」的程度
絕對值越大，代表該學生在「理科綜合能力」這個新維度上，距離平均水平越遠，表現越突出（無論是極好還是極差）。
學生F (+26.80)：他的絕對值最大，且是正數，代表他是這群人中理科綜合能力最強的。
學生A (-20.73)：他的絕對值也很大，但因為是負數，代表他是這群人中理科綜合能力最弱的。
學生D (+4.73) vs. 學生C (-4.88)：他們的絕對值都很小，代表他們兩位的理科綜合能力最接近平均水平。

3. 相對排序 (Ranking)：提供了新的單一維度排名
以前你需要比較兩個數字（數學和物理），現在你只需要比較這一個新分數，就可以對所有學生的理科表現進行線性排序。
從 F > E > D > C > B > A 這個排序，你可以一目了然地看出所有學生的理科綜合表現排名。
總結來說，這個投影後的數字，就是把原本二維平面上的點，拉到一條新的「數線」上。這條數線就是我們找到的最重要的主成分 (PC1)。每個點在這條新數線上的座標，就是它的新綜合分數。
"""

"""
簡單來說eigen vector裡面每一個數字就是loading, 
然後原始資料(去中心化的）dot eigen vector 就是最後PC scores？！
"""

"""
PC1, PC2, ..., PC_n are orthogonal (to each other)
主成分互相獨立 --> 獨立分析「理科綜合能力」(PC1) 和「文理偏差度」(PC2)，而不用擔心它們的意義互相糾纏。

"""