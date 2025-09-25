import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
EWMA (Exponentially Weighted Moving Average)
越靠近現在的數據，給予越大的權重（影響力）；越久遠的數據，權重就越小。
S_t = Y_t * 0.9 + (1-0.9) * S_t-1
我們來把它拆解一下：
S_t:今天 (時間點 t) 的 EWMA 預測值。這就是我們想求的結果。
Y_t:今天 (時間點 t) 的實際觀測值 (例如，今天實際的氣溫)。
S_t-1:昨天 (時間點 t-1) 的 EWMA 預測值。
a (alpha)：平滑常數 (Smoothing Constant)，一個介於 0 到 1 之間的數字。這是 EWMA 唯一的超參數，也是它的靈魂！
現在我們把公式翻譯成白話文：
今天的預測值 = (a * 今天的實際值) + ((1-a) * 昨天的預測值)
St​=αXt​+α(1−α)Xt−1​+α(1−α)2Xt−2​+…
alpha = 2/(span+1)

EWMA 的「現在預測值」其實不是要告訴你「未來幾天會多少」，
而是提供一個「去掉雜訊的基準線」，幫你看清真正的趨勢。順便預測
如果只有 Y_t
你只看到今天的雜訊。
如果只有 S_t-1
看到的是過去的平均，但可能跟不上變化。
把兩者結合，就能平滑雜訊又兼顧新訊號
smaller alpha, smoothier the curve (long term)

S_t+1=α⋅Y_t+1+(1−α)⋅S_t
EWMA 則像是一個靈活的追隨者。它的核心思想是「緊跟最近的數據」。數據有趨勢，它就跟著趨勢走；
數據沒趨勢，它就在原地徘徊。它不需要數據是平穩的，因為它的目標不是建立一個描述數據生成過程的複雜模型，而僅僅是為了平滑和追蹤數據。

# important
S_t+1 (Y_t+1) = EWMA_t

"""

# 1. 建立一個模擬的時間序列數據 (例如: 某商品每日銷售量)
np.random.seed(42)
# 建立一個基礎趨勢，再加上一些隨機波動
data = np.linspace(50, 150, 100) + np.random.randn(100) * 10 + 10 * np.sin(np.linspace(0, 20, 100))
dates = pd.date_range(start="2024-01-01", periods=100)
ts = pd.Series(data, index=dates)

# 2. 計算 EWMA
# 我們比較兩種不同的 alpha 值
# alpha=0.9: 高度重視近期數據，反應靈敏
ewma_sensitive = ts.ewm(alpha=0.9, adjust=False).mean()
# alpha=0.2: 更看重長期趨勢，結果更平滑
ewma_smooth = ts.ewm(alpha=0.2, adjust=False).mean()

# 3. 視覺化結果
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))
plt.plot(ts, label='原始銷售量 (Original Data)', color='gray', alpha=0.7)
plt.plot(ewma_sensitive, label='EWMA (alpha=0.9, 反應靈敏)', color='red', linestyle='--')
plt.plot(ewma_smooth, label='EWMA (alpha=0.2, 趨勢平滑)', color='blue')
plt.title('EWMA 效果展示', fontsize=16)
plt.legend()
plt.show()
