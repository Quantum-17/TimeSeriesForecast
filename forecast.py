import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Run this script to see the forecast for 2022.

# load dataset
daily_data=pd.read_csv('data_daily.csv', index_col=0, parse_dates=True)
daily_data.index.freq = 'D'

# create the model
#
# (p,d,q) = (2,2,1) 
#
# Setting the p, d, q values is the most signifcant configuration for this exercise.
# Arrived at these settings by running model_params.py for values to start with and
# then testing different combinations of values with model_test.py and forecast.py
# (this file) to choose the values that gave the best results.
#
# Please see model_params.py for how the initial values for p, d & q were chosen.
#
# Note the model is tested and trained using daily data even though we are interested
# in forecasting monthly data, as more datapoints for training will lead to better
# forecast results
#
model = ARIMA(daily_data, order=(2,2,1))
model_fit = model.fit()

# forecast for 2022
daily_forecast = model_fit.forecast(steps=365)

# print & plot monthly observed & forecasted
grouper = pd.Grouper(freq='ME')
observed_monthly = daily_data.groupby(grouper).sum()
forecasted_monthly = daily_forecast.groupby(grouper).sum().astype(int)

# convenience dataframe for printing & bar plot
combined_monthly = pd.DataFrame()
combined_monthly['Month'] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
combined_monthly['2021-Observed'] = observed_monthly['Receipt_Count'].values
combined_monthly['2022-Forecast'] = forecasted_monthly.values
combined_monthly.set_index('Month', inplace=True)

print()
print("-----------------------------------")
print("   Monthly Scanned Receipt Count   ")
print("-----------------------------------")
print(combined_monthly)
print("-----------------------------------")
print()

fig = plt.figure(figsize=(18,6), layout='constrained')
fig.suptitle('Scanned Receipt Count')
ax1 = plt.subplot2grid((2,2), (0,0), colspan=1)
ax2 = plt.subplot2grid((2,2), (0,1), colspan=1)
ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)

ax1.plot(observed_monthly, label='2021 Observed')
ax1.plot(forecasted_monthly, label='2022 Forecast')
ax1.set_title('Monthly Count')
ax1.set_ylabel('Receipt Count')
ax1.set_xticks(['2021-01-31', '2022-01-31', '2022-12-31'])
ax1.legend()

combined_monthly.plot.bar(title="Monthly Count 2021 vs. 2022", ax=ax2)
ax2.set_xlabel(None)

ax3.plot(daily_data, label='2021 Observed')
ax3.plot(daily_forecast, label='2022 Forecast')
ax3.set_title('Daily Count')
ax3.set_ylabel('Receipt Count')
ax3.set_xticks(['2021-01-01', '2022-01-01', '2022-12-31'])
ax3.legend()

plt.show()
