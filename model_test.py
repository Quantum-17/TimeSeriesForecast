import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Helper for testing the model by training on 2021 Jan-Aug data, then forecasting for
# 2021 Sep-Dec and comparing that with observed data for 2021 Sep-Dec.

# load dataset
daily_data=pd.read_csv('data_daily.csv', index_col=0, parse_dates=True)
daily_data.index.freq = 'D'

# split the daily data into training and testing data
num_jan_to_aug_days = 243 # there are 243 days from Jan 1 to Aug 31 in 2021 (non-leap year) 
train=daily_data.iloc[:num_jan_to_aug_days,:] # Jan-Aug
test=daily_data.iloc[num_jan_to_aug_days:,:] # Sep-Dec

# create the model
model = ARIMA(train, order=(2,2,1))
model_fit = model.fit()

# forecast for rest of the year, Sep-Dec 
daily_forecast = model_fit.forecast(steps=(365-num_jan_to_aug_days))

# print & plot monthly observed & forecasted
grouper = pd.Grouper(freq='ME')
observed_monthly = test.groupby(grouper).sum()
forecasted_monthly = daily_forecast.groupby(grouper).sum().astype(int)

# convenience dataframe for printing & bar plot
combined_monthly = pd.DataFrame()
combined_monthly['Month'] = ["Sep", "Oct", "Nov", "Dec"]
combined_monthly['Observed'] = observed_monthly['Receipt_Count'].values
combined_monthly['Forecast'] = forecasted_monthly.values
combined_monthly['Error %'] = ((combined_monthly['Forecast'] - combined_monthly['Observed'])/combined_monthly['Observed'])*100 
combined_monthly.set_index('Month', inplace=True)

print()
print("**************************************")
print("***** 2021 Scanned Receipt Count *****")
print("**************************************")
print()
print(combined_monthly)
print()
print("**************************************")
print()

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,6), layout='constrained')
fig.suptitle('Scanned Receipt Count')
ax1.plot(test, label='Observed')
ax1.plot(daily_forecast, label='Forecast')
ax1.set_title('2021 Sep-Dec Daily Observed vs. Forecast')
ax1.set_ylabel('Receipt Count')
ax1.legend()
ax1.set_xticks(['2021-09-01', '2021-12-31'])
combined_monthly.plot.bar(y=['Observed', 'Forecast'], title="2021 Sep-Dec Monthly Observed vs. Forecast", ax=ax2)
plt.show()
