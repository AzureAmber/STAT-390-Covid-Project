library(tidyverse)
library(tidymodels)
library(forecast)
library(timetk)
library(tictoc)
library(modeltime)
library(tseries)


# Read in data
train_lm <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm <- read_rds('data/avg_final_data/final_test_lm.rds')

# EDA - Plot the series, ACF, and PACF to help identify potential p, d, q
# Need to convert to time series obj
time_series <- train_lm_avg %>% select(new_cases) %>% as.ts()

# Plot the time series
train_lm |> 
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
stat_check <- tibble(country = character(23), 
                     adf = numeric(23), 
                     adf_pval = numeric(23), 
                     adf_state = character(23))

for (i in 1:23) {
  dat <- complete_lm %>% 
    filter(location == country_names[i]) %>%
    arrange(date)
  
  if (nrow(dat) > 0 && sum(!is.na(dat$new_cases)) > 0) {
    x <- ts(dat$new_cases, frequency = 7)
    y <- adf.test(x)
    
    stat_check$country[i] <- country_names[i]
    stat_check$adf[i] <- round(y$statistic, 2)
    stat_check$adf_pval[i] <- round(y$p.value, 2)
    stat_check$adf_state[i] <- ifelse(y$p.value <= 0.05, "Stationary", "Non-Stationary")
  } else {
    stat_check$country[i] <- country_names[i]
    stat_check$adf[i] <- NA
    stat_check$adf_pval[i] <- NA
    stat_check$adf_state[i] <- NA
  }
}

stat_check |> 
  print(n = 23)

# country          adf adf_pval adf_state     
# <chr>          <dbl>    <dbl> <chr>         
# 1 Argentina      -4.33     0.01 Stationary    
# 2 Australia      -2.03     0.56 Non-Stationary
# 3 Canada         -3.41     0.05 Non-Stationary
# 4 Colombia       -3.43     0.05 Non-Stationary
# 5 Ecuador        -4.09     0.01 Stationary    
# 6 Ethiopia       -4.59     0.01 Stationary    
# 7 France         -2.45     0.39 Non-Stationary
# 8 Germany        -2.52     0.36 Non-Stationary
# 9 India          -4.2      0.01 Stationary    
# 10 Italy          -2.84     0.23 Non-Stationary
# 11 Japan          -4.24     0.01 Stationary    
# 12 Mexico         -4.81     0.01 Stationary    
# 13 Morocco        -3.98     0.01 Stationary    
# 14 Pakistan       -4.37     0.01 Stationary    
# 15 Philippines    -3.74     0.02 Stationary    
# 16 Russia         -3.58     0.04 Stationary    
# 17 Saudi Arabia   -3.89     0.02 Stationary    
# 18 South Africa   -4.07     0.01 Stationary    
# 19 South Korea    -3.53     0.04 Stationary    
# 20 Sri Lanka      -2.47     0.38 Non-Stationary
# 21 Turkey         -3.45     0.05 Stationary    
# 22 United Kingdom -2.84     0.23 Non-Stationary
# 23 United States  -4.25     0.01 Stationary 

# Stationary Check Pt.2 ----
stat_check2 <- tibble(country = character(23), 
                      adf = numeric(23), 
                      adf_pval = numeric(23), 
                      adf_state = character(23))

for (i in 1:23) {
  dat <- train_lm_fix_init %>% 
    filter(location == country_names[i]) %>%
    arrange(date)
  
  if (nrow(dat) > 0 && sum(!is.na(dat$new_cases)) > 0) {
    x <- ts(dat$new_cases, frequency = 7)
    y <- adf.test(x)
    
    stat_check2$country[i] <- country_names[i]
    stat_check2$adf[i] <- round(y$statistic, 2)
    stat_check2$adf_pval[i] <- round(y$p.value, 2)
    stat_check2$adf_state[i] <- ifelse(y$p.value <= 0.05, "Stationary", "Non-Stationary")
  } else {
    stat_check2$country[i] <- country_names[i]
    stat_check2$adf[i] <- NA
    stat_check2$adf_pval[i] <- NA
    stat_check2$adf_state[i] <- NA
  }
}

stat_check2 |> 
  print(n = 23)

# country          adf adf_pval adf_state     
# <chr>          <dbl>    <dbl> <chr>         
# 1 Argentina      -3.9      0.02 Stationary    
# 2 Australia      -2.21     0.49 Non-Stationary
# 3 Canada         -3.18     0.09 Non-Stationary
# 4 Colombia       -3.09     0.12 Non-Stationary
# 5 Ecuador        -4.49     0.01 Stationary    
# 6 Ethiopia       -3.81     0.02 Stationary    
# 7 France         -2.49     0.37 Non-Stationary
# 8 Germany        -2.64     0.31 Non-Stationary
# 9 India          -3.37     0.06 Non-Stationary
# 10 Italy          -3.15     0.1  Non-Stationary
# 11 Japan          -4.62     0.01 Stationary    
# 12 Mexico         -4.31     0.01 Stationary    
# 13 Morocco        -3.74     0.02 Stationary    
# 14 Pakistan       -3.79     0.02 Stationary    
# 15 Philippines    -3.3      0.07 Non-Stationary
# 16 Russia         -3.35     0.07 Non-Stationary
# 17 Saudi Arabia   -3.42     0.05 Non-Stationary
# 18 South Africa   -3.67     0.03 Stationary    
# 19 South Korea    -3.53     0.04 Stationary    
# 20 Sri Lanka      -2.53     0.35 Non-Stationary
# 21 Turkey         -3.45     0.05 Stationary    
# 22 United Kingdom -2.77     0.26 Non-Stationary
# 23 United States  -3.95     0.01 Stationary  
####################### IGNORE CODE BELOW ##################

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
     
     