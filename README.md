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

Trains the model on the entire 2021 daily data and forecasts for the next 365 days (2022). Aggregates the daily forecast and observed counts into monthly counts.

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

Trains on 2021 Jan-Aug daily counts, then forecasts the 2021 Sep-Dec daily counts and compares against observed.

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

### Model Parameters and Evaluation
The crux of the solution lies in finding the right values for the p, d, q parameters for the ARIMA model. Running model_params.py is the starting point. I found this [article](https://analyticsindiamag.com/ai-mysteries/quick-way-to-find-p-d-and-q-values-for-arima/) incredibly useful for this part of the exercise, many thanks!

#### d param
From the graphs plotted by model_params.py we can see that the data becomes stationary after one order
of differencing, d = 1. However, d = 2 did better when testing the model by training on 2021 Jan-Aug
data, then forecasting for 2021 Sep-Dec and comparing that with the observed data for 2021 Sep-Dec.
d = 2 also gave better looking results when forecasting for 2022.

#### p param
From the PACF plots for both first and second order differencing we see that 2 lags cross the significance limit.
So we'll go with p = 2.

#### q param
From the ACF plots for both first and second order differencing we see that 2 lags cross the significance limit.
q = 2 indeed gave closer results during model testing and forecasting for 2021 Sep-Dec, however it
seemed to over-forecast for 2022, so went with q = 1 which gave better looking results for 2022
and close enough results for 2021 Sep-Dec.

#### Model Evaluation with different combinations of p, d, q
Our contenders are:
- (p = 2, d = 1, q = 2)
- (p = 2, d = 1, q = 1)
- (p = 2, d = 2, q = 2)
- (p = 2, d = 2, q = 1)
  
---
##### (p = 2, d = 1, q = 2) Results don't look good!

<pre>
---------------------------------------------
RMSE for 2021 Sep-Dec daily forecast = 563835
---------------------------------------------

-------------------------------------
      2021 Monthly Receipt Count
-------------------------------------
        Observed   Forecast   Error %
Month
Sep    281146154  277350014 -1.350237
Oct    295965185  286591369 -3.167202
Nov    296085162  277346486 -6.328813
Dec    309948684  286591369 -7.535865
-------------------------------------
</pre>

![Screenshot of model test results graphs.](/images/TestResult-2-1-2.png)
![Screenshot of forecast results graphs.](/images/Forecast-2-1-2.png)

<br>

---

##### (p = 2, d = 1, q = 1) Results don't look good, d = 1 is out!

<pre>
---------------------------------------------
RMSE for 2021 Sep-Dec daily forecast = 571004
---------------------------------------------

-------------------------------------
      2021 Monthly Receipt Count
-------------------------------------
        Observed   Forecast   Error %
Month
Sep    281146154  277075069 -1.448032
Oct    295965185  286315598 -3.260379
Nov    296085162  277079611 -6.418947
Dec    309948684  286315598 -7.624838
-------------------------------------
</pre>

![Screenshot of model test results graphs.](/images/TestResult-2-1-1.png)
![Screenshot of forecast results graphs.](/images/Forecast-2-1-1.png)

<br>

---

##### (p = 2, d = 2, q = 2) 2021 forecast vs. observed looks great, but 2022 seems to be over forecast

<pre>
---------------------------------------------
RMSE for 2021 Sep-Dec daily forecast = 257090
---------------------------------------------

-------------------------------------
      2021 Monthly Receipt Count
-------------------------------------
        Observed   Forecast   Error %
Month
Sep    281146154  278998535 -0.763880
Oct    295965185  298213374  0.759613
Nov    296085162  298209547  0.717491
Dec    309948684  318090739  2.626904
-------------------------------------
</pre>

![Screenshot of model test results graphs.](/images/TestResult-2-2-2.png)
![Screenshot of forecast results graphs.](/images/Forecast-2-2-2.png)

<br>

---

##### (p = 2, d = 2, q = 1) 2022 forecast looks better and 2021 forecast vs. observed is close enough, this is our winner!

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
![Screenshot of forecast results graphs.](/images/Forecast-2-2-1.png)


