library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(forecast)
library(lubridate)


# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/
# https://www.youtube.com/watch?v=8na-sasmu5I


         
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)


# Read in data
train_lm <- read_rds('data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')



train_lm %>% 
  select(date, new_cases) %>% 
  ggplot(aes(x = date, y = new_cases))+
  geom_line()

train_lm %>% 
  select(new_cases) %>% 
  ts() %>% 
  ggtsdisplay()

#ARIMA needs data to be stationary -> Dickey-Fuller test -> stationary

# train_lm %>% 
#   select(new_cases) %>% 
#   ts() %>% 
#   mstl() %>% 
#   autoplot()
# 
# train_lm %>% 
#   select(date, new_cases) %>% 
#   mutate(new_cases_daily_change = new_cases - lag(new_cases, 1),
#          lagged_new_cases = lag(new_cases, 1)) %>% 
#   ggplot(aes(x=date, y=new_cases_daily_change))+geom_line()

train_lm_2 <- train_lm %>% 
  mutate(new_cases_daily_change = new_cases - lag(new_cases, 1))

train_lm_2 %>% 
  select(new_cases_daily_change) %>% 
  ts() %>% 
  mstl() %>% 
  autoplot()


# with original linear model training data -> (5,1,0)
train_lm %>% 
  select(new_cases) %>% 
  ts() %>% 
  Arima(order = c(5,1,0)) %>% 
  forecast(h=12) %>% 
  autoplot()

# Ljung-Box test for original train
# 
# data:  Residuals from ARIMA(5,1,0)
# Q* = 1635.1, df = 5, p-value < 2.2e-16
# 
# Model df: 5.   Total lags used: 10


arima_model <- Arima(train_lm$new_cases, order = c(5,1,0))

summary(arima_model)

# Series: train_lm$new_cases 
# ARIMA(5,1,0) 
# 
# Coefficients:
#   ar1      ar2      ar3      ar4      ar5
# -0.5837  -0.5041  -0.3747  -0.2974  -0.1464
# s.e.   0.0062   0.0070   0.0073   0.0070   0.0062
# 
# sigma^2 = 4.453e+09:  log likelihood = -315249.1
# AIC=630510.2   AICc=630510.2   BIC=630559
# 
# Training set error measures:
#   ME     RMSE      MAE MPE MAPE      MASE        ACF1
# Training set 0.5801916 66723.41 23809.29 NaN  Inf 0.9970063 -0.02747603

ts_train_lm <- train_lm %>% 
  select(new_cases) %>% 
  ts(frequency = 366)

fitted_values <- fitted(arima_model)

plot(ts_train_lm, main = "ARIMA Model Fit vs Actual Data", ylab = "Values", xlab = "Time", col = "blue")

lines(fitted_values, col = "red")

legend("topleft", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1)






# with one lag -> (5,0,0)
train_lm_2 %>% 
  select(new_cases_daily_change) %>% 
  ts() %>% 
  Arima(order = c(5,0,0)) %>% 
  forecast(h=12) %>% 
  autoplot()

# Ljung-Box test
# 
# data:  Residuals from ARIMA(5,0,0) with non-zero mean
# Q* = 1635.3, df = 5, p-value < 2.2e-16
# 
# Model df: 5.   Total lags used: 10


arima_model_lag <- Arima(train_lm_2$new_cases_daily_change, order = c(5,0,0))

summary(arima_model_lag)

# Series: train_lm_2$new_cases_daily_change 
# ARIMA(5,0,0) with non-zero mean 
# 
# Coefficients:
#   ar1      ar2      ar3      ar4      ar5      mean
# -0.5838  -0.5041  -0.3747  -0.2975  -0.1464   -0.0442
# s.e.   0.0062   0.0070   0.0073   0.0070   0.0062  144.7295
# 
# sigma^2 = 4.453e+09:  log likelihood = -315249.1
# AIC=630512.2   AICc=630512.2   BIC=630569.1
# 
# Training set error measures:
#   ME     RMSE     MAE MPE MAPE      MASE        ACF1
# Training set 0.7085648 66724.74 23810.4 NaN  Inf 0.5501725 -0.02740888

ts_train_lm_lag <- train_lm_2 %>% 
  select(new_cases_daily_change) %>% 
  ts(frequency = 366)

fitted_values <- fitted(arima_model_lag)

plot(ts_train_lm_lag, main = "ARIMA Model Fit vs Actual Data", ylab = "Values", xlab = "Time", col = "blue")

lines(fitted_values, col = "red")

legend("topleft", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1)



##### some mutation for target variable (?)

train_lm %>% 
  mutate(new_cases = log10(new_cases)) %>% 
  ggplot(aes(x=date, y=log10(new_cases)))+
  geom_line()









############ keep the code below

# Check residuals for randomness, to ensure a good fit
checkresiduals(arima_model)




# Fit the ARIMA model 
# p = AR terms 
# d = differencing (to make ts stationary)
# q = MA terms
arima_model <- Arima(train_lm_2$new_cases_daily_change, order=c(5, 0, 0)) # white noise model




# Forecast
forecasted_values <- forecast(arima_model, h=10) # 'h' is the forecast horizon

# Plot forecast
plot(forecasted_values)





