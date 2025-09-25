import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 設定 matplotlib 樣式以獲得更好的視覺效果
plt.style.use('seaborn-v0_8-whitegrid')
# 讓中文字體能正常顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 生成模擬數據 ---

# 定義 ARIMA(2,1,1) 的真實參數
# 這些是我們在現實世界中需要去 "估計" 的值
# I 差分: Z_t = Y_t - Y_t-1
# Z_t=c+ϕ_1Z_t−1+ϕ_2Z_t−2+θ_1ϵ_t−1+ϵ_t 
ar_params_true = [0.5, 0.2]  # phi_1, phi_2
ma_params_true = [0.4]       # theta_1
c_true = 0.1                 # 常數項
n_points = 200               # 數據點數量

# AR(p), I(d), MA(q)
p = len(ar_params_true)
q = len(ma_params_true)
d = 1

# 生成白噪音 (white noise)
np.random.seed(42) # 為了可重複性
errors = np.random.normal(0, 1, n_points)

# 準備用來存放差分序列 Z_t 和原始序列 Y_t 的陣列
Z = np.zeros(n_points)
Y = np.zeros(n_points)

# 初始化前幾期的值 (因為模型需要參考過去的值)
Z[0:p] = errors[0:p]
Y[0] = 50 # 給定一個初始值

# 開始生成數據 (根據我們上面寫的數學式)
for t in range(p, n_points):
    # AR 部分
    ar_component = np.sum(ar_params_true * Z[t-p:t][::-1]) # [::-1] 反轉順序以匹配 Z_t-1, Z_t-2...
    
    # MA 部分
    ma_component = np.sum(ma_params_true * errors[t-q:t][::-1])
    
    # 計算差分序列 Z_t
    Z[t] = c_true + ar_component + ma_component + errors[t]
    
    # 從差分序列 Z_t 還原回原始序列 Y_t (I 的逆操作)
    # Z_t = Y_t - Y_{t-1}  =>  Y_t = Z_t + Y_{t-1}
    if t > 0:
        Y[t] = Z[t] + Y[t-1]

# 畫圖看看我們生成的數據
# plt.figure(figsize=(12, 6))
# plt.plot(Y, label='Original Yt')
# plt.title('ARIMA(2, 1, 1) data')
# plt.xlabel('time stamp')
# plt.ylabel('value')
# plt.legend()
# plt.show()

# exit()


def arima_manual_predict(series, ar_params, ma_params, c, d):
    """
    手刻的 ARIMA 預測函數 (一步預測)
    
    Args:
        series (np.array): 原始時間序列
        ar_params (list): AR 係數 [phi_1, phi_2, ...]
        ma_params (list): MA 係數 [theta_1, theta_2, ...]
        c (float): 常數項
        d (int): 差分階數
        
    Returns:
        tuple: (預測的原始序列值, 預測的差分序列值)
    """
    history = list(series)
    ar_params = np.array(ar_params)
    ma_params = np.array(ma_params)
    p = len(ar_params)
    q = len(ma_params)
    
    # --- Step I: 差分 ---
    # 根據 d 的值對歷史數據進行差分
    diff_history = np.diff(history, n=d)
    
    predictions_diff = []  # 存放差分序列的預測值 Z_hat
    predictions_orig = []  # 存放還原後原始序列的預測值 Y_hat
    errors = [0] * len(history) # 初始化誤差列表
    
    # 從第 p 個點開始，因為我們需要足夠的歷史數據來做 AR
    start_point = p
    
    for t in range(start_point, len(diff_history)):
        # --- Step ARMA: 預測差分序列 Z_t ---
        
        # 取得需要的歷史差分數據
        # Z_{t-1}, Z_{t-2}, ...
        ar_input = diff_history[t-p:t][::-1]
        
        # 取得需要的歷史誤差數據
        # epsilon_{t-1}, epsilon_{t-2}, ...
        ma_input = errors[t-q:t][::-1]

        # ARMA(p,q) 預測公式
        ar_term = np.sum(ar_params * ar_input)
        ma_term = np.sum(ma_params * ma_input)
        
        # 預測差分值
        z_hat = c + ar_term + ma_term
        predictions_diff.append(z_hat)
        
        # 計算真實的誤差
        # 這裡的 t 對應的是 diff_history 的 index
        z_true = diff_history[t]
        error = z_true - z_hat
        errors[t] = error
        
        # --- Step I (inverse): 還原預測值 ---
        # 預測 Y_hat_t = Z_hat_t + Y_{t-1}
        # series 的 index 比 diff_history 的 index 多 d
        # diff_history[t] 是 Y[t+d] - Y[t+d-1]
        # 所以對應的 Y_{t-1} 應該是 series[t+d-1]
        y_hat = z_hat + series[t+d-1]
        predictions_orig.append(y_hat)
        
    # 因為預測是從 start_point 開始的，前面補上 None 方便畫圖對齊
    # 原始序列的預測值，前面 p+d 個是無法預測的
    final_preds_orig = [None] * (p + d) + predictions_orig
        
    return final_preds_orig, predictions_diff

# --- 2. 使用手刻函數進行預測 ---
# 我們傳入 "真實" 的參數，來驗證我們的函數邏輯是否正確
predicted_Y, predicted_Z = arima_manual_predict(
    series=Y, 
    ar_params=ar_params_true, 
    ma_params=ma_params_true,
    c=c_true,
    d=d
)

# --- 3. 視覺化比較結果 ---
plt.figure(figsize=(15, 7))
plt.plot(Y, 'b-', label='原始數據 $Y_t$')
plt.plot(predicted_Y, 'r--', label='手刻 ARIMA 預測值 $\hat{Y}_t$', linewidth=2)
plt.title('手刻 ARIMA(2, 1, 1) 預測 vs. 原始數據')
plt.xlabel('時間點')
plt.ylabel('數值')
plt.legend()
plt.show()


# 建立並擬合模型
# 我們告訴模型我們的數據是 ARIMA(2,1,1) 結構
model = ARIMA(Y, order=(2, 1, 1))
model_fit = model.fit()

# 輸出模型摘要，看看它估計出的參數
print(model_fit.summary())

# 讓模型進行預測
# `start` 和 `end` 是原始數據的 index
# `dynamic=False` 表示使用一步預測 (in-sample prediction)，跟我們手刻的邏輯一樣
stats_preds = model_fit.predict(start=0, end=len(Y)-1, dynamic=False)

# 視覺化比較
plt.figure(figsize=(15, 7))
plt.plot(Y, 'b-', label='原始數據 $Y_t$')
plt.plot(predicted_Y, 'r--', label='手刻 ARIMA 預測值 $\hat{Y}_t$', linewidth=2.5)
plt.plot(stats_preds, 'g:', label='statsmodels 預測值', linewidth=2.5)
plt.title('手刻 vs. statsmodels ARIMA(2, 1, 1) 預測比較')
plt.xlabel('時間點')
plt.ylabel('數值')
plt.legend()
plt.show()