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


# 1. Read in data
train_lm <- read_rds('data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')

# 2. filter out the two countries we will be use

## stationary country rep
train_us <- train_lm %>% 
  filter(location == "United States")
test_us <- test_lm %>% 
  filter(location == "United States")

## non-stationary country rep
train_germ <- train_lm %>% 
  filter(location == "Germany")
test_germ <- test_lm %>% 
  filter(location == "Germany")

# 3. Create validation sets for every year train + 2 month test with 4-month increments
data_folds = rolling_origin(
  train_lm_fix,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds


# 4. automatic grid search for best p,d,q

# us: 3,1,5
train_us %>% 
  select(new_cases) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()

# germany:2,1,2
train_germ %>% 
  select(new_cases) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()

# 5. manual grid search for best p,d,q

p_range <- 0:5
d_range <- 0:2
q_range <- 0:5

best_aic <- Inf
best_order <- c(0,0,0)
best_model <- NULL

for (p in p_range){
  for(d in d_range){
    for(q in q_range){
      model <- tryCatch(Arima(train_germ, order = c(p,d,q)),
                        error = function(e) NULL)
      
      if(!is.null(model)){
        if (model$aic < best_aic){
          best_aic <- model$aic
          best_order <- c(p, d, q)
          best_model <- model
        }
      }
    }
  }
}


best_order
best_aic

forecast <- forecast(best_model, h = 12)

plot(forecast)

# 6. compare manual search and automatic search results by AIC

manual_model_us <- Arima(train_us$new_cases, order = c(0,0,0))
print(AIC(manual_model_us)) #28834.8

auto_model_us <- auto.arima(train_us$new_cases)
print(AIC(auto_model)) #26758.13


manual_model_germ <- Arima(train_germ$new_cases, order = c(0,0,0))
print(AIC(manual_model_germ)) #29248.58

auto_model_germ <- auto.arima(train_germ$new_cases)
print(AIC(auto_model_germ)) #28719.69

# 7. train the model using automatic search result (smaller AIC value)

arima_us <- arima_reg(
  seasonal_period = 12,
  non_seasonal_ar = 3, non_seasonal_differences = 1, non_seasonal_ma = 5,
  seasonal_ar = 1, seasonal_differences = tune(), seasonal_ma = 1) %>%
  set_engine('arima')

arima_recipe <- recipe(new_cases ~ date, data = train_us)

# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

arima_wflow <- workflow() %>%
  add_model(arima_us) %>%
  add_recipe(arima_recipe)

# 4. Setup tuning grid
arima_params <- arima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = seasonal_ar(c(0, 2)),
    seasonal_ma = seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid = grid_regular(arima_params, levels = 3)

# 5. Model Tuning
arima_tuned = tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)




############ hard code p, d, q #########################

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
  mutate(new_cases_daily_change = new_cases - dplyr::lag(new_cases, 1))

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

#train model with (5,1,0)
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

#plot fit vs actual data in training set
ts_train_lm <- train_lm %>% 
  select(new_cases) %>% 
  ts(frequency = 366)

fitted_values_normal <- fitted(arima_model)
plot(ts_train_lm, main = "ARIMA Model Fit vs Actual Data (Training)", ylab = "Values", xlab = "Time", col = "blue")
lines(fitted_values, col = "red")
legend("topleft", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1)

#forecast test set values using training model
forecasts_normal <- forecast(fitted_values_normal, h = 12)

#find RMSE of the test result
forecast_accuracy_normal <- forecast::accuracy(forecasts_normal, test_lm$new_cases)
rmse_value_normal <- forecast_accuracy_normal['Test set', 'RMSE']
print(rmse_value_normal) #21274.42

#plot fit vs actual data in test set
predicted_values <- forecasts_normal$mean

# Create a time series plot of the actual test values
plot(test_lm$new_cases, type = 'l', col = 'red', 
     ylim = range(c(test_lm$new_cases, predicted_values)), 
     ylab = "Values", xlab = "Time Index", 
     main = "Actual vs Forecasted Values (Test)")

# Add the forecasted values to the plot
lines(predicted_values, col = 'blue', type = 'l', lwd=2, lty=2)

# Add a legend to differentiate the lines
legend("topright", legend=c("Actual", "Forecasted"), col=c("red", "blue"), 
       lty=1:2, lwd=2, cex=0.8)





########### first differencing -> new_cases_daily_change (new target variable) ############


# t0 - t(-1) -> (5,0,0)
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

fitted_values_lag <- fitted(arima_model_lag)

plot(ts_train_lm_lag, main = "ARIMA Model Fit vs Actual Data", ylab = "Values", xlab = "Time", col = "blue")

lines(fitted_values, col = "red")

legend("topleft", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1)

test_lm_2 <- test_lm %>% 
  mutate(new_cases_daily_change = new_cases - dplyr::lag(new_cases, 1))

forecasts_lag <- forecast(fitted_values_lag, h = length(test_lm_2$new_cases_daily_change))


forecast_accuracy_lag <- forecast::accuracy(forecasts_lag, test_lm_2$new_cases_daily_change)
rmse_value_lag <- forecast_accuracy_lag['RMSE']

print(rmse_value_lag)

predicted_values_lag <- forecasts_lag$mean

# Create a time series plot of the actual test values
plot(test_lm_2$new_cases_daily_change, type = 'l', col = 'red', ylab = "Values", xlab = "Time Index", main = "Actual vs Forecasted Values (Test)")

# Add the forecasted values to the plot
lines(predicted_values_lag, col = 'blue', type = 'l')

# Add a legend to differentiate the lines
legend("topright", legend=c("Actual", "Forecasted"), col=c("red", "blue"), lty=1, cex=0.8)



##### some mutation for target variable (?)

train_lm %>% 
  mutate(new_cases = log10(new_cases)) %>% 
  ggplot(aes(x=date, y=log10(new_cases)))+
  geom_line()




















