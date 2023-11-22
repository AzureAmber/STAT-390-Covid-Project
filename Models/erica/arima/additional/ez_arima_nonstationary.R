library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(forecast)
library(lubridate)
library(tseries)


# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/
# https://www.youtube.com/watch?v=8na-sasmu5I



# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm <- read_rds('./data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('./data/finalized_data/final_test_lm.rds')

# 2. filter out non-stationary country rep

## non-stationary country rep
train_germ <- train_lm %>% 
  filter(location == "Germany")
test_germ <- test_lm %>% 
  filter(location == "Germany")

# 3. ARIMA can only work with stationary data

## try do first differencing to the data 

diff_train_germ <- diff(train_germ$new_cases)

diff_train_germ <- data.frame(
  date = tail(train_germ$date, -1),  # Exclude the first date because differencing reduces the length by 1
  new_cases_diff = diff(train_germ$new_cases)
)

# 4. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  diff_train_germ,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds




# adf_test <- tseries::adf.test(diff_train_germ$new_cases_diff, alternative = "stationary")
# 
# # Check the p-value -> 0.01 < 0.05, stationary now
# adf_test$p.value


arima_germ <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 2,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

# You will need to create a recipe for the differenced data
arima_recipe <- recipe(new_cases_diff ~ date, data = diff_train_germ)

# Create a workflow
arima_wflow <- workflow() %>%
  add_model(arima_germ) %>%
  add_recipe(arima_recipe)

save(arima_wflow, file = "Models/erica/results/arima_germ_wflow.rda" )


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
arima_tuned_germ <- tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)

arima_tuned_germ %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

save(arima_tuned_germ, file = "Models/erica/results/arima_tuned_germ_1.rda")

# 6. Results
arima_germ_autoplot <- autoplot(arima_tuned_germ, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_germ_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_germ_autoplot)
# Close the device
dev.off()

show_best(arima_tuned_germ, metric = "rmse") #RMSE: 38448

## BEST MODEL HYPERPARAMTER: p,d,q: 2,0,2; P,D,Q: 2,1,1


# 7. fit model

arima_wflow_tuned <- arima_wflow %>%
  finalize_workflow(select_best(arima_tuned_us, metric = "rmse"))

best_params <- select_best(arima_tuned_germ, metric = "rmse")

final_arima_model <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2,
  non_seasonal_differences = 0,
  non_seasonal_ma = 2,
  seasonal_ar = best_params$seasonal_ar,
  seasonal_differences = best_params$seasonal_differences,
  seasonal_ma = best_params$seasonal_ma
) %>% 
  set_engine("arima") %>% 
  fit(new_cases_diff ~ date, data = diff_train_germ)

# Once you've fitted the model, you can forecast with it as needed. 
# For instance, if you want to predict the next 30 days (assuming daily data):

# Forecast the next 30 days on the differenced scale
forecasts_diff <- predict(final_arima_model, new_data = tail(diff_train_germ, 30))

# Re-integrate the forecasts if necessary to get them on the original scale
# Note: ARIMA models automatically integrate the forecasts if they included differencing during fitting.
# However, since you have differenced the data manually, you will need to re-integrate manually as well.

# Get the last known value from the original (non-differenced) data
last_known_value <- tail(train_germ$new_cases, 1)

# Re-integrate the forecasts
forecasts_original_scale <- cumsum(c(last_known_value, forecasts_diff$.pred))[-1]

# Create a data frame of the re-integrated forecasts with the corresponding future dates
forecasts_df <- data.frame(
  date = seq.Date(max(train_germ$date) + 1, by = "day", length.out = length(forecasts_original_scale)),
  forecast = forecasts_original_scale
)

# Now you have 'forecasts_df' with forecasts on the original scale
In the example above, the final model is fitted with the fit() function from the parsnip package. It uses the best parameters found in the tuning process. We then generate predictions for the next 30 days based on the differenced data.

Since the differencing was done manually before fitting the model, you need to manually re-integrate the forecasts to transform them back to the original scale. This is done using the cumsum() function, but you have to exclude the first element because it’s the last actual known value from the original data and not a forecast.

Remember, in your case, since the data is manually differenced, the ARIMA model you fit should not include a differencing term (d=0). It assumes that you have already transformed the data into a stationary form.




User
i need this above message to include re-integrate forecast back to original scale as well, I did first differencing
ChatGPT
Certainly! To reintegrate the differenced forecasts back into the original scale, you would typically start with the last actual value from the original data and then add each differenced forecast to this last value cumulatively. Here's the modified part of the previous code where you re-integrate the forecasts:

r
Copy code
# Forecast the next 'n' periods on the differenced scale
n_forecast_periods <- 30
forecasts_diff <- predict(final_arima_model, new_data = new_data_frame(n = n_forecast_periods))

# Get the last known value from the original (non-differenced) data
last_known_value <- tail(train_germ$new_cases, 1)

# Re-integrate the forecasts
# Since the data was first-differenced, we add each forecasted difference to the original data
forecasts_original_scale <- last_known_value + cumsum(forecasts_diff$.pred)

# Create a data frame of the re-integrated forecasts with the corresponding future dates
forecasts_df <- data.frame(
  date = seq.Date(max(train_germ$date), by = "day", length.out = n_forecast_periods + 1),
  forecast = c(last_known_value, forecasts_original_scale)
)

# Now you have 'forecasts_df' with forecasts on the original scale
# Note: The first value in 'forecast' is the last actual known value, not a forecast
The predict function will give you the differenced forecasts. By adding the cumsum() of the differenced forecasts to the last known actual value (last_known_value), you "undifference" the forecasts. This gives you a forecast in the same scale as the original data.

Note that the new_data_frame function is a placeholder for creating a new data frame that the model will use for making predictions. You will need to create a function or a mechanism to generate this based on your specific requirements (typically a data frame with the same structure as the training data but with new dates).

The forecasts_df data frame will contain the undifferenced forecasts. The date column in the final forecasts dataframe should start with the day after the last known value and extend to the number of periods you are forecasting. The forecast column will start with the last known value (which is not a forecast) followed by the forecasted values on the original scale.






