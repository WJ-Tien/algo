
"""
information entropy and gini impurity (prob) 都望小
information gain (IG) 望大
一個分組（一個問題）是否完善，不是看劃分後子節點的 Gini 或 Entropy 是多少，
而是看劃分前後「不純度下降了多少」。這個下降的量，我們稱為「資訊增益 (Information Gain)」。

"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- 1. 準備資料 (Prepare Data) ---
# 這是經典的 "Play Tennis" 資料集
# 我們要根據天氣狀況，預測是否適合打網球 (Play)
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("--- 原始資料 ---")
print(df)
print("\n" + "="*30 + "\n")

# --- 2. 資料預處理 (Data Preprocessing) ---
# Scikit-learn 的決策樹模型需要數值型輸入，所以我們需要將類別特徵轉換為數字。
# 我們使用 One-Hot Encoding，因為這些特徵沒有順序性。

# 分離特徵 (X) 和目標 (y)
X = df.drop('Play', axis=1)
y = df['Play']

# 對特徵進行 One-Hot Encoding
# drop_first=True 可以避免共線性問題，同時減少特徵數量
X_encoded = pd.get_dummies(X, drop_first=True)

# 對目標變數進行 Label Encoding ('No' -> 0, 'Yes' -> 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("--- One-Hot 編碼後的特徵 (X) ---")
print(X_encoded.head()) # 只顯示前五行
print("\n--- Label 編碼後的目標 (y) ---")
print(f"Play 'No'  -> {le.transform(['No'])[0]}")
print(f"Play 'Yes' -> {le.transform(['Yes'])[0]}")
print(y_encoded)
print("\n" + "="*30 + "\n")


# --- 3. 建立與訓練模型 (Create and Train the Model) ---
# 為了讓範例簡單並能視覺化整棵樹，我們這裡直接用全部資料來訓練
# 在真實專案中，你會需要用 train_test_split 來劃分訓練集和測試集

# 建立決策樹分類器
# criterion='gini': 使用 Gini Impurity 作為不純度的衡量標準。你也可以改成 'entropy'
# max_depth=3: 限制樹的最大深度為 3，這是一個防止過擬合的常用方法。
# random_state=42: 確保每次執行的結果都一樣，方便重現。
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)


# 使用 Gini Impurity 訓練模型
print("--- 使用 Gini Impurity 訓練模型 ---")
model_gini.fit(X_encoded, y_encoded)
print("模型訓練完成！")

# 使用 Information Entropy 訓練模型
print("\n--- 使用 Information Entropy 訓練模型 ---")
model_entropy.fit(X_encoded, y_encoded)
print("模型訓練完成！")
print("\n" + "="*30 + "\n")


# --- 4. 視覺化決策樹 (Visualize the Decision Tree) ---
# 視覺化可以幫助我們理解模型是如何做決策的

# 視覺化 Gini Tree
plt.figure(figsize=(20, 10))
plot_tree(model_gini,
          feature_names=X_encoded.columns,
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Decision Tree Trained with GINI Impurity (使用 Gini Impurity 訓練的決策樹)", fontsize=20)
# plt.savefig("decision_tree_gini.png") # 如果需要，可以取消註解來存檔
plt.show()

# 視覺化 Entropy Tree
plt.figure(figsize=(20, 10))
plot_tree(model_entropy,
          feature_names=X_encoded.columns,
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Decision Tree Trained with Information Entropy (使用 Entropy 訓練的決策樹)", fontsize=20)
# plt.savefig("decision_tree_entropy.png") # 如果需要，可以取消註解來存檔
plt.show()


# --- 5. 進行預測 (Make a Prediction) ---
# 讓我們來預測一筆新的天氣資料
# 天氣: 晴天 (Sunny), 溫度: 適中 (Mild), 濕度: 高 (High), 風: 強 (Strong)
new_data = {
    'Outlook_Sunny': [1],
    'Outlook_Overcast': [0], # 因為原始資料有 Overcast，所以 get_dummies 會產生這個欄位
    'Temperature_Hot': [0],
    'Temperature_Mild': [1],
    'Humidity_Normal': [0], # High 是 drop_first=True 被丟掉的基準，所以 Normal=0 代表 High
    'Wind_Weak': [0] # Strong 是 drop_first=True 被丟掉的基準，所以 Weak=0 代表 Strong
}
# 確保欄位順序與訓練時一致
new_df = pd.DataFrame(new_data, columns=X_encoded.columns)

print("--- 預測新資料 ---")
print("新資料內容:")
print(new_df)

prediction_code = model_gini.predict(new_df)
prediction_proba = model_gini.predict_proba(new_df)
prediction_text = le.inverse_transform(prediction_code)

print(f"\n模型預測結果 (代碼): {prediction_code[0]}")
print(f"模型預測結果 (文字): {prediction_text[0]}")
print(f"預測機率 ('No', 'Yes'): {prediction_proba[0]}")

# 從樹的圖中，你可以手動跟著流程走一遍，驗證這個預測結果！
# 1. Outlook_Sunny <= 0.5?  (No, it's 1, go Right)
# 2. Humidity_Normal <= 0.5? (Yes, it's 0, go Left) -> value = [2, 0], class = No
# 所以模型預測 "No"，這和我們的訓練資料中 Outlook=Sunny, Humidity=High 的情況一致。
