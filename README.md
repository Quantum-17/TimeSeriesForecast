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
- model_test.py - script to test the model, trains on 2021 Jan-Aug data, forecasts for 2021 Sep-Dec and compares against observed
- model_params.py - script to help determine the p, d, q values for the ARIMA model
#### Modules Required
- pandas
- matplotlib.pyplot
- statsmodels.tsa.arima.model
- scikit-learn
- statsmodels.graphics.tsaplots (needed only if you are running model_params.py)
- statsmodels.tsa.stattools (needed only if you are running model_params.py)
#### Run the forecast
```console 
python3 forecast.py
```

You should see the following print out and graph:
<pre>
-----------------------------------
   Monthly Scanned Receipt Count
-----------------------------------
       2021-Observed  2022-Forecast
Month                              
Jan        236736687      321027070
Feb        220033460      297377066
Mar        248608625      337441806
Apr        250644830      334763984
May        263151748      354403760
Jun        260656840      351178777
Jul        274776003      371365713
Aug        283943231      379985722
Sep        281146154      375935515
Oct        295965185      396947676
Nov        296085162      392350309
Dec        309948684      413909629
-----------------------------------
</pre>

![Screenshot of forecast results graphs.](/images/Forecast-2-2-1.png)

#### Run the model test
```console 
python3 model_test.py
```

You should see the following print out and graph:
<pre>
---------------------------------------------
RMSE for 2021 Sep-Dec daily forecast = 399715
---------------------------------------------

-------------------------------------
      2021 Monthly Receipt Count     
-------------------------------------
        Observed   Forecast   Error %
Month                                
Sep    281146154  276015732 -1.824824
Oct    295965185  288971605 -2.362974
Nov    296085162  283261058 -4.331221
Dec    309948684  296434582 -4.360109
-------------------------------------
</pre>

![Screenshot of model test results graphs.](/images/TestResult-2-2-1.png)
