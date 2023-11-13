library(tidyverse)
library(tidymodels)
library(forecast)
library(timetk)
library(tictoc)
library(modeltime)
library(tseries)

tidymodels_prefer()

# Read in data
train_lm <- read_rds('data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')

# EDA - Plot the series, ACF, and PACF to help identify potential p, d, q
# Need to convert to time series obj
time_series <- train_lm %>% select(new_cases) %>% as.ts()

# Plot the time series
train_lm |> 
  filter(location == "Germany") |> 
  select(date, new_cases) |> 
  plot_time_series(date, new_cases, .smooth = FALSE, .title = "Time Series Plot for Germany") 


# Stationarity
adf.test(time_series)

# Plot ACF and PACF
Acf(time_series)
Pacf(time_series)

# Fit the ARIMA model 
  # p = AR terms 
  # d = differencing (to make ts stationary)
  # q = MA terms
arima_model <- Arima(time_series, order=c(0, 0, 0)) # white noise model

# ASK ABOUT THIS ABOVE!!

# Check residuals for randomness, to ensure a good fit
checkresiduals(arima_model)

# Ljung-Box test
# 
# data:  Residuals from ARIMA(0,0,0) with non-zero mean
# Q* = 2669.8, df = 10, p-value < 2.2e-16
# 
# Model df: 0.   Total lags used: 10

# Forecast
forecasted_values <- forecast(arima_model, h=10) # 'h' is the forecast horizon

# Plot forecast
plot(forecasted_values)
     
     