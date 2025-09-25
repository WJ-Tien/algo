import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

"""
Indirect: S_t-2 -> S_t-1 -> S_t
Direct: S_t-2 -> S_t
ACF: indirect and direct effect
PACF: direct effect only

ACF:
p_k = Cov(Y_t, Y_t-k) / Var(Yt)
Var: 序列自身的變異數，用於將共變異數標準化到 [-1, 1] 的區間。
"""


# --- 範例 A：特徵極度明顯的 AR(1) 模型 ---
# Y_t = 0.9 * Y_{t-1} + epsilon_t
ar_params_strong = np.array([1, -0.9]) # AR(1) 參數，phi=0.9
ma_params_strong = np.array([1])      # 無 MA 部分
ar_process_strong = ArmaProcess(ar_params_strong, ma_params_strong)
ar_data_strong = ar_process_strong.generate_sample(nsample=500)

# --- 繪製 AR(1) 的圖表 ---
fig_ar, axes_ar = plt.subplots(1, 3, figsize=(18, 5))
axes_ar[0].plot(ar_data_strong)
axes_ar[0].set_title('Strong AR(1) Process Data (phi=0.9)')
plot_acf(ar_data_strong, ax=axes_ar[1], lags=25)
axes_ar[1].set_title('ACF of Strong AR(1) Process')
plot_pacf(ar_data_strong, ax=axes_ar[2], lags=25, method='ywm')
axes_ar[2].set_title('PACF of Strong AR(1) Process')
fig_ar.tight_layout()
plt.show()