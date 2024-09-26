import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Helper script for choosing the p, d & q parameters for our model.
# 
# d param
# -------
# From the graphs plotted by this script we can see that the data becomes stationary after one order
# of differencing, d = 1. However, d = 2 did better when testing the model by training on 2021 Jan-Aug
# data, then forecasting for 2021 Sep-Dec and comparing that with observed data for 2021 Sep-Dec.
# d = 2 also gave better looking results when forecasting for 2022.
#
# p param
# -------
# From the PACF plots for both first and second order differencing we see that 2 lags cross the
# significance limit. So we'll go with p = 2.
#
# q param
# -------
# From the ACF plots for both first and second order differencing we see that 2 lags cross the
# significance limit. q = 2 indeed gave closer results during model testing and forecasting for
# 2021 Sep-Dec, however it seemed to over-forecast for 2022, so went with q = 1 which gave better
# looking results for 2022 and close enough results for 2021 Sep-Dec.
#

# load dataset
daily_data=pd.read_csv('data_daily.csv', index_col=0, parse_dates=True)
daily_data.index.freq = 'D'

result = adfuller(daily_data.dropna())
print('Daily data is', 'stationary' if result[1] < 0.05 else 'non-stationary', 'p-value:', result[1])
result = adfuller(daily_data.diff().dropna())
print('Daily data after 1st order differencing is', 'stationary' if result[1] < 0.05 else 'non-Stationary', 'p-value:', result[1])
result = adfuller(daily_data.diff().diff().dropna())
print('Daily data after 2nd order differencing is', 'stationary' if result[1] < 0.05 else 'non-Stationary', 'p-value:', result[1])

fig, axs = plt.subplots(8, figsize=(18,12), layout='constrained')

axs[0].set_title("Daily data")
axs[0].plot(daily_data)

axs[1].set_title("Daily data with 1st order differencing")
axs[1].plot(daily_data.diff())

axs[2].set_title("Daily data with 2nd order differencing")
axs[2].plot(daily_data.diff().diff())

plot_acf(daily_data.dropna(), ax=axs[3])

plot_acf(daily_data.diff().dropna(), ax=axs[4])
axs[4].set_title("Autocorelation with 1st order differencing")

plot_acf(daily_data.diff().diff().dropna(), ax=axs[5])
axs[5].set_title("Autocorelation with 2nd order differencing")

plot_pacf(daily_data.diff().dropna(), ax=axs[6])
axs[6].set_title("Partial Autocorelation with 1st order differencing")

plot_pacf(daily_data.diff().diff().dropna(), ax=axs[7])
axs[7].set_title("Partial Autocorelation with 2nd order differencing")

plt.show()

