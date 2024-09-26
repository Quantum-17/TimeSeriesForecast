# Time Series Forecasting
My first exercise in time series forecasting. I had fun learning about it and implementing a solution.
## Problem
Forecast total scanned receipts for each month in year 2022 given the number of scanned receipts for each day in 2021
## Solution
My solution uses ARIMA. After reading up on the various options such as ARIMA, SARIMA and LSTM, I decided to start with the simplest approach using ARIMA. It gave reasonablly good results.
### How to run the solution
#### Files
- data_daily.csv - contains the number of scanned receipts for each day in 2021
- forecast.py - script to run the forecast for 2022
- model_test.py - script to test the model, trains on 2021 Jan-Aug data, forecasts for 2021 Sep-Dec and compares it against the observed data for 2021 Sep-Dec
- model_params.py - script to help determine the p, d, q values for the ARIMA model
#### Modules Required
- pandas
- matplotlib.pyplot
- statsmodels.tsa.arima.model
- statsmodels.graphics.tsaplots (needed only if you are running model_params.py)
- statsmodels.tsa.stattools (needed only if you are running model_params.py)
#### Run the forecast
- python3 forecast.py
#### Run the test
- python3 model_test.py 
