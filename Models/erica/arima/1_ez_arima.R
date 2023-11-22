library(tidyverse) 
library(tidymodels)
library(modeltime)
library(doParallel)
library(forecast)
library(lubridate)
library(RcppRoll)


tidymodels_prefer()

# 1. Read in data

train_lm <- readRDS('data/avg_final_data/final_train_lm.rds')
test_lm <- readRDS('data/avg_final_data/final_test_lm.rds')

# weekly rolling average of new cases
complete_lm <- train_lm %>% rbind(test_lm) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), new_cases, value)) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(which(row_number() %% 7 == 0)) %>%
  mutate(
    time_group = row_number(),
    seasonality_group = row_number() %% 53) %>%
  ungroup() %>%
  mutate(seasonality_group = as.factor(seasonality_group))

# remove observations before COVID even started

complete_lm %>% 
  select(location, date, new_cases) %>% 
  filter(date < as.Date("2023-01-01")) %>% 
  filter(new_cases == 0) %>% 
  view()

#looks like South Korea is the country that started to have COVID cases the earliest; so we remove
#all observations before 2020/01/23 since those would not be meaningful

complete_lm_update <- complete_lm %>% 
  filter(date > as.Date("2020-01-23"))

# Split into train and test

train_lm <- complete_lm_update %>% filter(date < as.Date("2023-01-01")) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()
test_lm <- complete_lm_update %>% filter(date >= as.Date("2023-01-01")) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()



# 2. Find model trend by country
train_lm_fix <- NULL
test_lm_fix <- NULL
country_names <- unique(train_lm$location)
for (i in country_names) {
  data = train_lm %>% filter(location == i)
  complete_data = complete_lm_update %>% filter(location == i)
  # find linear model by country
  lm_model = lm(value ~ 0 + time_group + seasonality_group,
                data %>% filter(between(time_group, 13, nrow(data) - 12)))
  x = complete_data %>%
    mutate(
      trend = predict(lm_model, newdata = complete_data),
      slope = as.numeric(coef(lm_model)["time_group"]),
      seasonality_add = trend - slope * time(date),
      err = value - trend) %>%
    mutate_if(is.numeric, round, 5)
  train_lm_fix <<- rbind(train_lm_fix, x %>% filter(date < as.Date("2023-01-01")))
  test_lm_fix <<- rbind(test_lm_fix, x %>% filter(date >= as.Date("2023-01-01")))
}



# save train and test data by country in separate dataframe


for (loc in country_names) {
  location_data <- train_lm_fix %>% filter(location == loc)
  location_name <- paste0("train_lm_fix_", make.names(loc))  # Create the name with "train_" prefix
  assign(location_name, location_data, envir = .GlobalEnv)
}

for (loc in country_names) {
  location_data <- test_lm_fix %>% filter(location == loc)
  location_name <- paste0("test_lm_fix_", make.names(loc))  # Create the name with "train_" prefix
  assign(location_name, location_data, envir = .GlobalEnv)
}


# plot of original data and trend
ggplot(train_lm_fix_United.States) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, trend), color = 'red')

# plot of residual errors
x %>% filter(location == "United States") %>% 
  ggplot(aes(date, err))+
  geom_line()



# ARIMA Model tuning for errors
# 3. Create validation sets for every year train + 2 month test with 4-month increments

data_folds <- rolling_origin(
  train_lm_fix,
  initial = 53,
  assess = 4*2,
  skip = 4*4,
  cumulative = FALSE
)



# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_United.States %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 2,0,2

arima_model <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 2,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe <- recipe(err ~ date, data = train_lm_fix_United.States)


#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow <- workflow() %>%
  add_model(arima_model) %>%
  add_recipe(arima_recipe)



# 5. Setup tuning grid
arima_params <- arima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid <- grid_regular(arima_params, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)

arima_tuned <- tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

save(arima_tuned, file = "Models/erica/results/arima_tuned.rda")

stopCluster(cores.cluster)

arima_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot <- autoplot(arima_tuned, metric = "rmse")
show_best(arima_tuned, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot)
# Close the device
dev.off()

### Tuning results is (2,0,2), (2,0,0) ###


# 7. fit train and predict test


arima_wflow_tuned <- arima_wflow %>%
  finalize_workflow(select_best(arima_tuned, metric = "rmse"))

arima_fit_us <- fit(arima_wflow_tuned, train_lm_fix_United.States)

final_train_us <- train_lm_fix_United.States %>%
  bind_cols(pred_err = arima_fit_us$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# error model
final_train_us %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = err, color = "train_actual")) + 
  geom_line(aes(y = pred_err, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "Error Model (US)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

# prediction model

us_train_pred <- final_train_us %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Training: Actual vs. Predicted New Cases in United States",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,2), (P,D,Q) = (2,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(us_train_pred, file = "Results/erica/arima/us_train_pred.jpeg")

library(ModelMetrics)
# rmse of error prediction
ModelMetrics::rmse(final_train_us$err, final_train_us$pred_err)
# rmse of just linear trend
ModelMetrics::rmse(final_train_us$value, final_train_us$trend)
# rmse of linear trend + arima
ModelMetrics::rmse(final_train_us$value, final_train_us$pred) #20443.79


# Testing set
final_test_us <- test_lm_fix_United.States %>%
  bind_cols(predict(arima_fit_us, new_data = test_lm_fix_United.States)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

# initial prediction with just linear trend

final_test_us %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "test_actual")) + 
  geom_line(aes(y = trend, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "Linear Trend Prediction ONLY - Testing (US)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

# final prediction with linear trend + arima error modelling



us_test_pred <- final_test_us %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "1 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in United States",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,2), (P,D,Q) = (2,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(us_test_pred, file = "Results/erica/arima/us_test_pred.jpeg")


# rmse of just linear trend
ModelMetrics::rmse(final_test_us$value, final_test_us$trend)
# rmse of linear trend + arima
ModelMetrics::rmse(final_test_us$value, final_test_us$pred) #162069.4




#### fit train and predict test using the same model


# for loop to fit train and predict test

# Initialize an empty tibble to store RMSE results
rmse_results <- tibble(country = character(), 
                       rmse_pred_train = numeric(), 
                       rmse_pred_test = numeric())

for (location in country_names) {
  ## Replace spaces with dots for the data frame names
  location_df_name <- gsub(" ", "\\.", location)
  
  # Construct data frame names for train and test sets
  train_data_name <- paste0("train_lm_fix_", location_df_name)
  test_data_name <- paste0("test_lm_fix_", location_df_name)
  
  # Use tryCatch to handle errors
  tryCatch({
    # Check if the train data frame exists
    if (exists(train_data_name, where = .GlobalEnv)) {
      # Access the train and test data frames using get()
      train_data <- get(train_data_name)
      test_data <- get(test_data_name)
    
    # Fit the ARIMA model
    arima_fit <- fit(arima_wflow_tuned, get(train_data_name))
    
    # Prepare training data
    final_train <- get(train_data_name) %>%
      bind_cols(pred_err = arima_fit$fit$fit$fit$data$.fitted) %>%
      mutate(pred = trend + pred_err) %>%
      mutate_if(is.numeric, round, 5)
    
    # RMSE calculations for training data
    rmse_err_train <- ModelMetrics::rmse(final_train$err, final_train$pred_err)
    rmse_trend_train <- ModelMetrics::rmse(final_train$value, final_train$trend)
    rmse_pred_train <- ModelMetrics::rmse(final_train$value, final_train$pred)
    
    # Prepare testing data
    final_test <- get(test_data_name) %>%
      bind_cols(predict(arima_fit, new_data = get(test_data_name))) %>%
      rename(pred_err = .pred) %>%
      mutate(pred = trend + pred_err) %>%
      mutate_if(is.numeric, round, 5)
    
    # RMSE calculations for testing data
    rmse_trend_test <- ModelMetrics::rmse(final_test$value, final_test$trend)
    rmse_pred_test <- ModelMetrics::rmse(final_test$value, final_test$pred)
    
    # Rest of your code for processing and calculating RMSE...
    
    # Append results to the tibble
    rmse_results <- rbind(rmse_results, 
                          tibble(country = location, 
                                 rmse_pred_train = rmse_pred_train, 
                                 rmse_pred_test = rmse_pred_test))
    } else {
      message("Data frame not found for ", location)
    }
  }, error = function(e) {
    message("Error fitting model for ", location, ": ", e$message)
  })
}

print(rmse_results)
write.csv(rmse_results, "rmse_results.csv", row.names = FALSE)










for(country in country_names){
  # Create plot for the current country
  plot_name <- paste("Training: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/prophet_multi/training_plots/prophet_multi_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  prophet_multi_country <- ggplot(prophet_train_results %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 0, changepoint_range = 0.6,
         prior_scale_changepoints = 100, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.001)",
         caption = "Prophet Multivariate",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, prophet_multi_country, width = 10, height = 6)
}


## Error fitting model: non-finite finite-difference value:
## Canada, Morocco, Phillippines, South Africa

## Argentina

# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Argentina %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 3,0,2

arima_model_arg <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 3, non_seasonal_differences = 0, non_seasonal_ma = 2,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_arg <- recipe(err ~ date, data = train_lm_fix_Argentina)


#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow_arg <- workflow() %>%
  add_model(arima_model_arg) %>%
  add_recipe(arima_recipe_arg)



# 5. Setup tuning grid
arima_params_arg <- arima_wflow_arg %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_arg <- grid_regular(arima_params_arg, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_arg <- tune_grid(
  arima_wflow_arg,
  resamples = data_folds,
  grid = arima_grid_arg,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

save(arima_tuned_arg, file = "Models/erica/results/arima_tuned_arg.rda")

stopCluster(cores.cluster)

arima_tuned_arg %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot_arg <- autoplot(arima_tuned_arg, metric = "rmse")
show_best(arima_tuned_arg, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_arg.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_arg)
# Close the device
dev.off()

# 7. fit train and predict test

arima_wflow_tuned_arg <- arima_wflow_arg %>%
  finalize_workflow(select_best(arima_tuned_arg, metric = "rmse"))

arima_fit_arg <- fit(arima_wflow_tuned_arg, train_lm_fix_Argentina)

final_train_arg <- train_lm_fix_Argentina %>%
  bind_cols(pred_err = arima_fit_arg$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
final_train_arg %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (Argentina)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_arg$value, final_train_arg$pred) #3652.457


# Testing set
final_test_arg <- test_lm_fix_Argentina %>%
  bind_cols(predict(arima_fit_arg, new_data = test_lm_fix_Argentina)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

final_test_arg %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "test_actual")) + 
  geom_line(aes(y = pred, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "Linear Trend + arima Testing (Argentina)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_arg$value, final_test_arg$pred) #18566.12



### Australia
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Australia %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 2,1,1

arima_model_aus <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 1, non_seasonal_ma = 1,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_aus <- recipe(err ~ date, data = train_lm_fix_Australia)


#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow_aus <- workflow() %>%
  add_model(arima_model_aus) %>%
  add_recipe(arima_recipe_aus)



# 5. Setup tuning grid
arima_params_aus <- arima_wflow_aus %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_aus <- grid_regular(arima_params_aus, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_aus <- tune_grid(
  arima_wflow_aus,
  resamples = data_folds,
  grid = arima_grid_aus,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

save(arima_tuned_aus, file = "Models/erica/results/arima_tuned_aus.rda")

stopCluster(cores.cluster)

arima_tuned_aus %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot_aus <- autoplot(arima_tuned_aus, metric = "rmse")
show_best(arima_tuned_aus, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_aus.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_aus)
# Close the device
dev.off()

# 7. fit train and predict test

arima_wflow_tuned_aus <- arima_wflow_aus %>%
  finalize_workflow(select_best(arima_tuned_aus, metric = "rmse"))

arima_fit_aus <- fit(arima_wflow_tuned_aus, train_lm_fix_Australia)

final_train_aus <- train_lm_fix_Australia %>%
  bind_cols(pred_err = arima_fit_aus$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
final_train_aus %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (Australia)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_aus$value, final_train_aus$pred) #3537.556


# Testing set
final_test_aus <- test_lm_fix_Australia %>%
  bind_cols(predict(arima_fit_aus, new_data = test_lm_fix_Australia)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

final_test_aus %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "test_actual")) + 
  geom_line(aes(y = pred, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "Linear Trend + arima Testing (Australia)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_aus$value, final_test_aus$pred) #27560.46




### Germany
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Germany %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 3,0,0

arima_model_ger <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 3, non_seasonal_differences = 0, non_seasonal_ma = 0,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_ger <- recipe(err ~ date, data = train_lm_fix_Germany)


#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow_ger <- workflow() %>%
  add_model(arima_model_ger) %>%
  add_recipe(arima_recipe_ger)



# 5. Setup tuning grid
arima_params_ger <- arima_wflow_ger %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_ger <- grid_regular(arima_params_ger, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_ger <- tune_grid(
  arima_wflow_ger,
  resamples = data_folds,
  grid = arima_grid_ger,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

save(arima_tuned_ger, file = "Models/erica/results/arima_tuned_ger.rda")

load("Models/erica/results/arima/arima_tuned_ger.rda")

stopCluster(cores.cluster)

arima_tuned_ger %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot_ger <- autoplot(arima_tuned_ger, metric = "rmse")

show_best(arima_tuned_ger, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_ger.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_ger)
# Close the device
dev.off()

# 7. fit train and predict test

arima_wflow_tuned_ger <- arima_wflow_ger %>%
  finalize_workflow(select_best(arima_tuned_ger, metric = "rmse"))

arima_fit_ger <- fit(arima_wflow_tuned_ger, train_lm_fix_Germany)

final_train_ger <- train_lm_fix_Germany %>%
  bind_cols(pred_err = arima_fit_ger$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
ger_train_pred <- final_train_ger %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (Germany)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

ggsave(ger_train_pred, file = "Results/erica/arima/ger_train_pred.jpeg")


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_ger$value, final_train_ger$pred) #5770.618


# Testing set
final_test_ger <- test_lm_fix_Germany %>%
  bind_cols(predict(arima_fit_ger, new_data = test_lm_fix_Germany)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

ger_test_pred <- final_test_ger %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "test_actual")) + 
  geom_line(aes(y = pred, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "Linear Trend + arima Testing (Germany)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

ggsave(ger_test_pred, file = "Results/erica/arima/ger_test_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_ger$value, final_test_ger$pred) #106454


### Phillipines
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Philippines %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 3,1,1

arima_model_phi <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 3, non_seasonal_differences = 1, non_seasonal_ma = 1,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_phi <- recipe(err ~ date, data = train_lm_fix_Philippines)


#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow_phi <- workflow() %>%
  add_model(arima_model_phi) %>%
  add_recipe(arima_recipe_phi)



# 5. Setup tuning grid
arima_params_phi <- arima_wflow_phi %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_phi <- grid_regular(arima_params_phi, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_phi <- tune_grid(
  arima_wflow_phi,
  resamples = data_folds,
  grid = arima_grid_phi,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

save(arima_tuned_phi, file = "Models/erica/results/arima_tuned_phi.rda")

stopCluster(cores.cluster)

arima_tuned_phi %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot_phi <- autoplot(arima_tuned_phi, metric = "rmse")

show_best(arima_tuned_phi, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_phi.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_phi)
# Close the device
dev.off()

# 7. fit train and predict test

arima_wflow_tuned_phi <- arima_wflow_phi %>%
  finalize_workflow(select_best(arima_tuned_phi, metric = "rmse"))

arima_fit_phi <- fit(arima_wflow_tuned_phi, train_lm_fix_Philippines)

final_train_phi <- train_lm_fix_Philippines %>%
  bind_cols(pred_err = arima_fit_phi$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
final_train_phi %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (Phillipines)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_phi$value, final_train_phi$pred) #1492.395


# Testing set
final_test_phi <- test_lm_fix_Philippines %>%
  bind_cols(predict(arima_fit_phi, new_data = test_lm_fix_Philippines)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

final_test_phi %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "test_actual")) + 
  geom_line(aes(y = pred, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "Linear Trend + arima Testing (Phillipines)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_phi$value, final_test_phi$pred) #4742.375


### Saudi Arabia
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Saudi.Arabia %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 1,1,1

arima_model_saudi <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 1, non_seasonal_differences = 1, non_seasonal_ma = 1,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_saudi <- recipe(err ~ date, data = train_lm_fix_Saudi.Arabia)


#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow_saudi <- workflow() %>%
  add_model(arima_model_saudi) %>%
  add_recipe(arima_recipe_saudi)



# 5. Setup tuning grid
arima_params_saudi <- arima_wflow_saudi %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_saudi <- grid_regular(arima_params_saudi, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_saudi <- tune_grid(
  arima_wflow_saudi,
  resamples = data_folds,
  grid = arima_grid_saudi,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

save(arima_tuned_saudi, file = "Models/erica/results/arima_tuned_saudi.rda")

stopCluster(cores.cluster)

arima_tuned_saudi %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot_saudi <- autoplot(arima_tuned_saudi, metric = "rmse")

show_best(arima_tuned_saudi, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_saudi.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_saudi)
# Close the device
dev.off()

# 7. fit train and predict test

arima_wflow_tuned_saudi <- arima_wflow_saudi %>%
  finalize_workflow(select_best(arima_tuned_saudi, metric = "rmse"))

arima_fit_saudi <- fit(arima_wflow_tuned_saudi, train_lm_fix_Saudi.Arabia)

final_train_saudi <- train_lm_fix_Saudi.Arabia %>%
  bind_cols(pred_err = arima_fit_saudi$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
final_train_saudi %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (Saudi Arabia)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_saudi$value, final_train_saudi$pred) #224.0634


# Testing set
final_test_saudi <- test_lm_fix_Saudi.Arabia %>%
  bind_cols(predict(arima_fit_saudi, new_data = test_lm_fix_Saudi.Arabia)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

final_test_saudi %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "test_actual")) + 
  geom_line(aes(y = pred, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "Linear Trend + arima Testing (Saudi Arabia)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_saudi$value, final_test_saudi$pred) #1103.035



### South Africa
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()


arima_fit_safrica <- fit(arima_wflow_tuned_saudi, train_lm_fix_South.Africa)


final_train_safrica <- train_lm_fix_South.Africa %>%
  bind_cols(pred_err = arima_fit_safrica$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
final_train_safrica %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (South Africa)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_safrica$value, final_train_safrica$pred) #1310.274


# Testing set

final_test_safrica <- test_lm_fix_South.Africa %>%
  bind_cols(predict(arima_fit_safrica, new_data = test_lm_fix_South.Africa)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

final_test_safrica %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "test_actual")) + 
  geom_line(aes(y = pred, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "Linear Trend + arima Testing (South Africa)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_safrica$value, final_test_safrica$pred) #6863.421






