# 7. fit train and predict test

arima_wflow_tuned <- arima_wflow %>%
  finalize_workflow(select_best(arima_tuned_germ, metric = "rmse"))

diff_data <- diff(train_germ$new_cases)

adf_test <- adf.test(diff_data, alternative = "stationary")

# Check the p-value
adf_test$p.value


arima_fit <- fit(arima_wflow_tuned, diff_data)

#training set prediction & graph
train_predictions <- predict(arima_fit, new_data = train_us) %>%
  bind_cols(train_us %>% select(date, new_cases)) %>%
  mutate(estimate = .pred) %>%
  select(date, new_cases, estimate)

train_predictions %>%
  yardstick::rmse(new_cases, estimate) #RMSE: 133255

train_predictions %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = new_cases, color = "train_actual")) +
  geom_line(aes(y = estimate, color = "train_pred"), linetype = "dashed") +
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data",
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (US)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


#test set prediction & graph

test_predictions <- predict(arima_fit, new_data = test_us) %>%
  bind_cols(test_us %>% select(date, new_cases))%>%
  mutate(estimate = .pred) %>%
  select(date, new_cases, estimate)

test_predictions %>%
  yardstick::rmse(new_cases, estimate) #RMSE: 71678

test_predictions %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = new_cases, color = "test_actual")) +
  geom_line(aes(y = estimate, color = "test_pred"), linetype = "dashed") +
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data",
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (US)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


## for all the other countries that data are stationary

# first extract countries
locations <- unique(train_lm$location)

for (loc in locations) {
  location_data <- train_lm %>% filter(location == loc)
  location_name <- paste0("train_", make.names(loc))  # Create the name with "train_" prefix
  assign(location_name, location_data, envir = .GlobalEnv)
}

for (loc in locations) {
  location_data <- test_lm %>% filter(location == loc)
  location_name <- paste0("test_", make.names(loc))  # Create the name with "train_" prefix
  assign(location_name, location_data, envir = .GlobalEnv)
}

###### use the same hyperparamter to fit other countries' data

#non-stationary countries, so we remove those
countries_of_interest <- c("Australia", "France", "Germany",
                           "Japan", "Sri Lanka", "Turkey")

locations <- setdiff(unique(train_lm$location), countries_of_interest)

fitted_models <- list()

# Loop through each location, fit the model, and store the result
for (loc in locations) {
  
  train_df_name <- paste0("train_", make.names(loc))
  
  if (exists(train_df_name)) {
    
    train_data <- get(train_df_name)
    
    fitted_model <- fit(arima_wflow_tuned, data = train_data)
    
    fitted_models[[loc]] <- fitted_model
  }
}


# Prepare a list to store the RMSE values for each location
rmse_values <- list()

# Loop through each fitted model and calculate predictions on the training set
for (loc in names(fitted_models)) {
  
  fitted_model <- fitted_models[[loc]]
  
  train_df_name <- paste0("train_", make.names(loc))
  
  if (exists(train_df_name)) {
    
    train_data <- get(train_df_name)
    
    train_predictions <- predict(fitted_model, new_data = train_data) %>%
      bind_cols(train_data %>% select(date, new_cases)) %>%
      mutate(estimate = .pred) %>%
      select(date, new_cases, estimate)
    
    rmse_value <- train_predictions %>%
      yardstick::rmse(truth = new_cases, estimate = estimate)
    
    rmse_values[[loc]] <- rmse_value
  }
}

rmse_tibble <- tibble(
  location = names(rmse_values),
  rmse = sapply(rmse_values, function(rmse_df) { rmse_df$.estimate })
)

print(rmse_tibble) %>%
  arrange(rmse)


# location          rmse
# <chr>            <dbl>
#   1 Ethiopia          742.
# 2 Saudi Arabia     1290.
# 3 Ecuador          1582.
# 4 Pakistan         2198.
# 5 Morocco          2245.
# 6 South Africa     6236.
# 7 Canada           6644.
# 8 Philippines      6699.
# 9 Colombia         8943.
# 10 Mexico          10093.
# 11 Argentina       20050.
# 12 Russia          33104.
# 13 Italy           34479.
# 14 United Kingdom  35621.
# 15 South Korea     74888.
# 16 India           83228.
# 17 United States  133255.
















