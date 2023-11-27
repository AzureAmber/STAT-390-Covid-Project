library(tidyverse)
library(tidymodels)
library(forecast)
library(timetk)
library(tictoc)
library(modeltime)
library(tseries)


# Read in data
train_lm_avg <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm_avg <- read_rds('data/avg_final_data/final_test_lm.rds')
train_lm <- read_rds('data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')

# EDA - Plot the series, ACF, and PACF to help identify potential p, d, q
# Need to convert to time series obj
time_series <- train_lm_avg %>% select(new_cases) %>% as.ts()

# Plot the time series
train_lm_avg |> 
  filter(location == "United States") |> 
  select(date, new_cases) |> 
  plot_time_series(date, 
                   new_cases, 
                   .smooth = FALSE, 
                   .title = "Time Series Plot for United States",
                   .line_color = "black",
                   .x_intercept_color = "black",
                   .y_intercept_color = "black",
                   .x_lab = "Date", 
                   .y_lab = "New Cases") 


# Stationarity Test ----
# Using the new avg_final_data with weekly averages
adf_results_avg <- train_lm_avg |> 
  group_by(location) |> 
  summarize(
    adf_test = list(adf.test(new_cases, alternative = 'stationary')),
    adf_statistic = adf_test[[1]]$statistic,
    pvalue = adf_test[[1]]$p.value,
    stationary = adf_test[[1]]$p.value < 0.05
  ) |> 
  ungroup()

adf_results_avg |> 
  select(location, adf_statistic, pvalue, stationary) |> 
  print(n = 23)

# location       adf_statistic pvalue stationary
# <chr>                  <dbl>  <dbl> <lgl>     
# 1 Argentina              -5.28 0.01   TRUE      
# 2 Australia              -3.91 0.0134 TRUE      
# 3 Canada                 -5.82 0.01   TRUE      
# 4 Colombia               -3.81 0.0186 TRUE      
# 5 Ecuador                -4.16 0.01   TRUE      
# 6 Ethiopia               -5.00 0.01   TRUE      
# 7 France                 -6.02 0.01   TRUE      
# 8 Germany                -4.11 0.01   TRUE      
# 9 India                  -5.18 0.01   TRUE      
# 10 Italy                  -5.63 0.01   TRUE      
# 11 Japan                  -3.17 0.0932 FALSE     
# 12 Mexico                 -6.31 0.01   TRUE      
# 13 Morocco                -4.37 0.01   TRUE      
# 14 Pakistan               -4.03 0.01   TRUE      
# 15 Philippines            -6.10 0.01   TRUE      
# 16 Russia                 -5.98 0.01   TRUE      
# 17 Saudi Arabia           -3.79 0.0193 TRUE      
# 18 South Africa           -4.08 0.01   TRUE      
# 19 South Korea            -4.43 0.01   TRUE      
# 20 Sri Lanka              -2.05 0.556  FALSE     
# 21 Turkey                 -4.10 0.01   TRUE      
# 22 United Kingdom         -3.53 0.0398 TRUE      
# 23 United States          -6.18 0.01   TRUE  

# Using previous finalized_data
adf_results_og <- train_lm |> 
  group_by(location) |> 
  summarize(
    adf_test = list(adf.test(new_cases, alternative = 'stationary')),
    ADF_Statistic = adf_test[[1]]$statistic,
    P_Value = adf_test[[1]]$p.value,
    Stationary = adf_test[[1]]$p.value < 0.05
  ) |> 
  ungroup()

non_stationary_countries <- adf_results_avg |> 
  filter(!Stationary) |> 
  pull(location)
  # Japan and Sri Lanka are non-stationary for avg_final_data

adf_results_og |> 
  filter(!Stationary) |> 
  pull(location)
  # Australia, France, Germany, Japan, Sri Lanka, Turkey are non-stationary for og final data

# Looking at distribution of Japan and Sri Lanka
train_lm |> 
  filter(location == "Japan") |> 
  ggplot(aes(new_cases)) + 
  geom_density() + 
  theme_minimal()

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
     
     