"""
ARIMA(p, d, q) 是什麼？
當我們把這三者組合起來，就成了一個強大的 ARIMA 模型：
p (AutoRegressive part)：我們要用過去多少期的歷史觀測值來預測未來？
d (Integrated part, 差分)：我們要對數據做幾次差分才能讓它變平穩？
q (Moving Average part)：我們要用過去多少期的歷史預測誤差來預測未來？

AR(Auto Regressive, p, 慣性):  
    Y_t=c+ϕ_1Y_t−1+ϕ_2Y_t−2+...+ϵ_t
    Y_t：今天的數值。
    Y_t−1,Y_t−2：昨天、前天的數值。
    ϕ_1,ϕ_2：權重係數，代表昨天的值對今天的影響有多大。
    ϵ_t：今天的隨機誤差（無法被模型解釋的白噪音）。
    ARIMA 中的 p (past values) 參數，就是指我們要考慮過去多少天的數值。例如 p=2 就是說，今天的數值跟昨天、前天的數值都有關。
    現在你已經得到了「體溫變化量」這個序列。接下來，你會考慮病人的體質，也就是慣性 (Inertia)。
    你知道這位病人體質虛弱，如果他今天體溫上升了，那麼明天很可能也會繼續上升，因為他的身體有這種「慣性」。反之，如果他今天體溫下降了，明天也很可能繼續下降。
    這就是 AR (自我迴歸) 的精神！它假設今天的狀態，會受到昨天、前天...等過去狀態的直接影響。就像物理學的慣性定律，運動中的物體會傾向於保持運動。
    AR(p) 中的 p 就是問：我們要考慮過去幾天的「慣性」？p=1 就是只看昨天的影響，p=2 就是看昨天和前天的綜合影響。

I (intergrated/differencing, d, 變化): 
    差分 (Differencing) 的精神！我們不關心數值的絕對高低，我們關心的是它的變化
    差分，就是把焦點從「狀態」轉移到「狀態的變化」，這是讓不穩定問題變穩定的第一步。
    Y'_t=Y_t−Y_t−1

MA (moving average, q, random/error): 
    這個模型假設今天的數值，跟過去幾天的「預測誤差」有關。也就是說，模型會吸取過去犯錯的教訓。
    生活化例子：想像你在玩一個射飛鏢的遊戲，你每次都瞄準紅心，但總會有點偏差（這就是誤差）。
    如果你發現你連續幾次都射偏在紅心的左邊，下一次你瞄準時，就會下意識地往右邊修正一點。
    你修正的依據，就是基於你過去的誤差。MA 模型就是捕捉這種「對過去誤差的修正」機制。
    Y_t=μ+ϵ_t+θ_1ϵ_t−1+θ_2ϵ_t−2+...
    Y_t：今天的數值。
    ϵ_t,ϵ_t−1：今天、昨天的預測誤差。
    θ_1,θ_2：權重係數，代表昨天的誤差對今天的影響有多大。
    ARIMA 中的 q (past errors) 參數，就是指我們要考慮過去多少天的預測誤差。
    例如 q=1 就是說，今天的數值會受到昨天預測誤差的影響。


ARIMA 非常強大，但它不是萬靈丹。它有自己的「能力範圍」：
它假設線性關係：ARIMA 本質上是一個線性模型。如果數據中的關係非常複雜、非線性（例如複雜的週期性、混沌現象），ARIMA 的效果會很差。
對「黑天鵝」事件無能為力：ARIMA 依賴於歷史數據的模式。如果未來發生了一個從未有過的突發事件（例如 COVID-19 疫情爆發、公司政策突然轉變），模型是無法預測的。
不擅長處理變動的波動性：有些時間序列（尤其是金融數據），它的波動幅度本身會隨時間改變（例如牛市時波動小，熊市時波動大）。ARIMA 假設波動性（變異數）是穩定的，處理這種問題需要更進階的 ARCH/GARCH 模型。
無法納入外部變數：ARIMA 只看數據本身。但有時候，我們要預測的目標會受到其他因素影響。例如，預測冰淇淋銷量，如果能把「天氣溫度」這個外部變數加進來，模型肯定會更準。純粹的 ARIMA 做不到這點（但它的擴展版 ARIMAX 可以）。

Y_t+1 = Z_t + Y_t
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller # 用來檢驗平穩性

# 1. 建立一個模擬的、有趨勢的時間序列數據
np.random.seed(0)
n_samples = 150
# 建立一個線性趨勢和一些AR(1)特性
ar_params = np.array([0.75])
ma_params = np.array([])
data = [100]
for i in range(1, n_samples):
    # y(t) = 0.75 * y(t-1) + error + 0.5 (趨勢)
    new_val = data[i-1] * ar_params[0] + np.random.randn() + 0.5
    data.append(new_val)

dates = pd.date_range(start="2024-01-01", periods=n_samples)
ts = pd.Series(data, index=dates)

# 我們的數據有明顯的上升趨勢，所以它是不平穩的
# 我們可以透過 ADF 檢定來確認 (p-value > 0.05 代表不平穩)
result = adfuller(ts)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}') # p-value 很大，確認不平穩

# 2. 建立並訓練 ARIMA 模型
# 我們知道數據有趨勢，所以 d 至少為 1
# 我們假設 p=1 (因為數據是跟前一期相關), q=0
# 所以模型是 ARIMA(1, 1, 0)
model = ARIMA(ts, order=(1, 1, 0))
model_fit = model.fit()

# 打印模型摘要
print(model_fit.summary())

# 3. 進行預測
# 分割數據，用前120筆來訓練，預測後30筆
train = ts[:120]
test = ts[120:]

model = ARIMA(train, order=(1, 1, 0))
model_fit = model.fit()

# 預測未來30期
forecast = model_fit.forecast(steps=30)

# 4. 視覺化結果
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Train data')
plt.plot(test.index, test, label='Real data (Test)', color='orange')
plt.plot(forecast.index, forecast, label='ARIMA (Forecast)', color='red', linestyle='--')
plt.title('ARIMA predict', fontsize=16)
plt.legend()
plt.show()