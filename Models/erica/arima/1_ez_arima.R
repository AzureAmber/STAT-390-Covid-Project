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


arima_model_us <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 2,
  seasonal_ar = 2, seasonal_differences = 0, seasonal_ma = 0) %>%
  set_engine('arima')

arima_recipe_us <- recipe(err ~ date, data = train_lm_fix_United.States)


arima_wflow_tuned <- workflow() %>%
  add_model(arima_model_us) %>%
  add_recipe(arima_recipe_us)


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
    
    # Check if the final_train and final_test data frames are available
    if (exists("final_train") && exists("final_test")) {
      # Plot for training data
      train_plot <- ggplot(final_train, aes(x=date)) +
        geom_line(aes(y = value, color = "Actual New Cases")) +
        geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
        scale_y_continuous(n.breaks = 15) + 
        scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
        theme_minimal() + 
        labs(x = "Date", 
             y = "New Cases", 
             title = paste0("Training: Actual vs. Predicted New Cases in ", location),
             subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,2), (P,D,Q) = (2,0,0))",
             caption = "ARIMA",
             color = "") + 
        theme(plot.title = element_text(face = "bold", hjust = 0.5),
              plot.subtitle = element_text(face = "italic", hjust = 0.5),
              legend.position = "bottom",
              panel.grid.minor = element_blank()) +
        scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
      
      # Plot for testing data
      test_plot <- ggplot(final_test, aes(x=date)) +
        geom_line(aes(y = value, color = "Actual New Cases")) +
        geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
        scale_y_continuous(n.breaks = 15) + 
        scale_x_date(date_breaks = "1 months", date_labels = "%b %y") +
        theme_minimal() + 
        labs(x = "Date", 
             y = "New Cases", 
             title = paste0("Testing: Actual vs. Predicted New Cases in ", location),
             subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,2), (P,D,Q) = (2,0,0))",
             caption = "ARIMA",
             color = "") + 
        theme_light()+
        theme(plot.title = element_text(face = "bold", hjust = 0.5),
              plot.subtitle = element_text(face = "italic", hjust = 0.5),
              legend.position = "bottom",
              panel.grid.minor = element_blank()) +
        scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
      
      ggsave(train_plot, file = paste0("Results/erica/arima/", location,"_train_pred",  ".jpeg"),
             width=8, height =7, dpi = 300)
      ggsave(test_plot, file = paste0("Results/erica/arima/", location, "_test_pred", ".png"),
             width=8, height = 7, dpi =300)
      
    } else {
      message("Final train/test data not available for ", location)
    }
    
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
write.csv(rmse_results, "Results/erica/arima/rmse_results.csv", row.names = FALSE)




## Error fitting model: non-finite finite-difference value:
## Australia, Canada, Morocco, Philippines, South Africa


### Australia
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Australia %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 2,1,2

arima_model_aus <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 1, non_seasonal_ma = 2,
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


stopCluster(cores.cluster)

arima_tuned_aus %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot_aus <- autoplot(arima_tuned_aus, metric = "rmse")
show_best(arima_tuned_aus, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima/arima_autoplot_aus.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_aus)
# Close the device
dev.off()

### (2,1,2), (0,0,0)

# 7. fit train and predict test

arima_model_aus <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 1, non_seasonal_ma = 2,
  seasonal_ar = 0, seasonal_differences = 0, seasonal_ma = 0) %>%
  set_engine('arima')

arima_recipe_aus <- recipe(err ~ date, data = train_lm_fix_Australia)

arima_wflow_tuned_aus <- workflow() %>%
  add_model(arima_model_aus) %>%
  add_recipe(arima_recipe_aus)


arima_fit_aus <- fit(arima_wflow_tuned_aus, train_lm_fix_Australia)

final_train_aus <- train_lm_fix_Australia %>%
  bind_cols(pred_err = arima_fit_aus$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model

Australia_train_pred <- final_train_aus %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Training: Actual vs. Predicted New Cases in Australia",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,1,2), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Australia_train_pred, file = "Results/erica/arima/Australia_train_pred.jpeg")


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_aus$value, final_train_aus$pred) #2528.537


# Testing set
final_test_aus <- test_lm_fix_Australia %>%
  bind_cols(predict(arima_fit_aus, new_data = test_lm_fix_Australia)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

Australia_test_pred <- final_test_aus %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "1 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in Australia",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,1,2), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Australia_test_pred, file = "Results/erica/arima/Australia_test_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_aus$value, final_test_aus$pred) #23330.13




### Canada
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Canada %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 1,0,2

arima_model_can <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 1, non_seasonal_differences = 0, non_seasonal_ma = 2,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_can <- recipe(err ~ date, data = train_lm_fix_Canada)

arima_wflow_can <- workflow() %>%
  add_model(arima_model_can) %>%
  add_recipe(arima_recipe_can)


# 5. Setup tuning grid
arima_params_can <- arima_wflow_can %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_can <- grid_regular(arima_params_can, levels = 3)

# 6. Model Tuning
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_can <- tune_grid(
  arima_wflow_can,
  resamples = data_folds,
  grid = arima_grid_can,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)


stopCluster(cores.cluster)

arima_tuned_can %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
arima_autoplot_can <- autoplot(arima_tuned_can, metric = "rmse")

show_best(arima_tuned_can, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_can.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_can)
# Close the device
dev.off()

# 8. fit train and predict test

### (1,0,2), (0,0,0)

arima_model_can <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 1, non_seasonal_differences = 0, non_seasonal_ma = 2,
  seasonal_ar = 0, seasonal_differences = 0, seasonal_ma = 0) %>%
  set_engine('arima')

arima_recipe_can <- recipe(err ~ date, data = train_lm_fix_Canada)

arima_wflow_tuned_can <- workflow() %>%
  add_model(arima_model_can) %>%
  add_recipe(arima_recipe_can)


arima_fit_can <- fit(arima_wflow_tuned_can, train_lm_fix_Canada)

final_train_can <- train_lm_fix_Canada %>%
  bind_cols(pred_err = arima_fit_can$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model

Canada_train_pred <- final_train_can %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Training: Actual vs. Predicted New Cases in Canada",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (1,0,2), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Canada_train_pred, file = "Results/erica/arima/Canada_train_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_train_can$value, final_train_can$pred) #1188.261


# Testing set
final_test_can <- test_lm_fix_Canada %>%
  bind_cols(predict(arima_fit_can, new_data = test_lm_fix_Canada)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

Canada_test_pred <- final_test_can %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in Canada",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (1,0,2), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Canada_test_pred, file = "Results/erica/arima/Canada_test_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_can$value, final_test_can$pred) #8333.362


### Phillipines
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Philippines %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 2,1,2

arima_model_phi <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 1, non_seasonal_ma = 2,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_phi <- recipe(err ~ date, data = train_lm_fix_Philippines)


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


# 8. fit train and predict test

### (2,1,2), (0,0,0)

arima_model_phi <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 1, non_seasonal_ma = 2,
  seasonal_ar = 0, seasonal_differences = 0, seasonal_ma = 0) %>%
  set_engine('arima')

arima_recipe_phi <- recipe(err ~ date, data = train_lm_fix_Philippines)

arima_wflow_tuned_phi <- workflow() %>%
  add_model(arima_model_phi) %>%
  add_recipe(arima_recipe_phi)

arima_fit_phi <- fit(arima_wflow_tuned_phi, train_lm_fix_Philippines)

final_train_phi <- train_lm_fix_Philippines %>%
  bind_cols(pred_err = arima_fit_phi$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


Phillipines_train_pred <- final_train_phi %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Training: Actual vs. Predicted New Cases in Phillipines",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,1,2), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Phillipines_train_pred, file = "Results/erica/arima/Phillipines_train_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_train_phi$value, final_train_phi$pred) #1519.059


# Testing set
final_test_phi <- test_lm_fix_Philippines %>%
  bind_cols(predict(arima_fit_phi, new_data = test_lm_fix_Philippines)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

Phillipines_test_pred <- final_test_phi %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "1 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in Phillipines",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,1,2), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Phillipines_test_pred, file = "Results/erica/arima/Phillipines_test_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_phi$value, final_test_phi$pred) #4625.488


### Morroco
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_Morocco %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 2,0,1

arima_model_morroco <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 1,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_morroco <- recipe(err ~ date, data = train_lm_fix_Morocco)


#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow_morroco <- workflow() %>%
  add_model(arima_model_morroco) %>%
  add_recipe(arima_recipe_morroco)



# 5. Setup tuning grid
arima_params_morroco <- arima_wflow_morroco %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_morroco <- grid_regular(arima_params_morroco, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_morroco <- tune_grid(
  arima_wflow_morroco,
  resamples = data_folds,
  grid = arima_grid_morroco,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)

arima_tuned_morroco %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results

arima_autoplot_morroco <- autoplot(arima_tuned_morroco, metric = "rmse")

show_best(arima_tuned_morroco, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_morroco.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_morroco)
# Close the device
dev.off()



# 8. fit train and predict test

### (2,0,1), (1,0,0)

arima_model_morocco <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 1,
  seasonal_ar = 1, seasonal_differences = 0, seasonal_ma = 0) %>%
  set_engine('arima')

arima_recipe_morocco <- recipe(err ~ date, data = train_lm_fix_Morocco)

arima_wflow_tuned_morocco <- workflow() %>%
  add_model(arima_model_morocco) %>%
  add_recipe(arima_recipe_morocco)


arima_fit_morocco <- fit(arima_wflow_tuned_morocco, train_lm_fix_Morocco)

final_train_morocco <- train_lm_fix_Morocco %>%
  bind_cols(pred_err = arima_fit_morocco$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
Morocco_train_pred <- final_train_morocco %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Training: Actual vs. Predicted New Cases in Morocco",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,1), (P,D,Q) = (1,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Morocco_train_pred, file = "Results/erica/arima/Morocco_train_pred.jpeg")


# rmse of linear trend + arima
ModelMetrics::rmse(final_train_morocco$value, final_train_morocco$pred) #355.0955


# Testing set
final_test_morocco <- test_lm_fix_Morocco %>%
  bind_cols(predict(arima_fit_morocco, new_data = test_lm_fix_Morocco)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# testing visualization

Morocco_test_pred <- final_test_morocco %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "1 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in Morocco",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,1), (P,D,Q) = (1,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(Morocco_test_pred, file = "Results/erica/arima/Morocco_test_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_morocco$value, final_test_morocco$pred) #1449.738


### South Africa
# 4. Define model, recipe, and workflow

## determine p, d, q by looking at auto.arima()

train_lm_fix_South.Africa %>% 
  select(err) %>% 
  ts() %>% 
  auto.arima() %>% 
  summary()  # 3,0,3

arima_model_safrica <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 3, non_seasonal_differences = 0, non_seasonal_ma = 3,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe_safrica <- recipe(err ~ date, data = train_lm_fix_South.Africa)


arima_wflow_safrica <- workflow() %>%
  add_model(arima_model_safrica) %>%
  add_recipe(arima_recipe_safrica)



# 5. Setup tuning grid
arima_params_safrica <- arima_wflow_safrica %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = non_seasonal_ar(c(0, 2)),
    seasonal_ma = non_seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid_safrica <- grid_regular(arima_params_safrica, levels = 3)

# 6. Model Tuning
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)

arima_tuned_safrica <- tune_grid(
  arima_wflow_safrica,
  resamples = data_folds,
  grid = arima_grid_safrica,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)

arima_tuned_safrica %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results

arima_autoplot_safrica <- autoplot(arima_tuned_safrica, metric = "rmse")

show_best(arima_tuned_safrica, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_autoplot_safrica.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_autoplot_safrica)
# Close the device
dev.off()



# 8. fit train and predict test

### (3,0,3), (0,0,0)

arima_model_safrica <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 3, non_seasonal_differences = 0, non_seasonal_ma = 3,
  seasonal_ar = 0, seasonal_differences = 0, seasonal_ma = 0) %>%
  set_engine('arima')

arima_recipe_safrica <- recipe(err ~ date, data = train_lm_fix_South.Africa)

arima_wflow_tuned_safrica <- workflow() %>%
  add_model(arima_model_safrica) %>%
  add_recipe(arima_recipe_safrica)


arima_fit_safrica <- fit(arima_wflow_tuned_safrica, train_lm_fix_South.Africa)


final_train_safrica <- train_lm_fix_South.Africa %>%
  bind_cols(pred_err = arima_fit_safrica$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# prediction model
safrica_train_pred <- final_train_safrica %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Training: Actual vs. Predicted New Cases in South Africa",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (3,0,3), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(safrica_train_pred, file = "Results/erica/arima/South Africa_train_pred.jpeg")



# rmse of linear trend + arima
ModelMetrics::rmse(final_train_safrica$value, final_train_safrica$pred) #1179.209


# Testing set

final_test_safrica <- test_lm_fix_South.Africa %>%
  bind_cols(predict(arima_fit_safrica, new_data = test_lm_fix_South.Africa)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# final prediction with linear trend + arima error modelling

# testing visualization

safrica_test_pred <- final_test_safrica %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = value, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "1 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in South Africa",
       subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (3,0,3), (P,D,Q) = (0,0,0))",
       caption = "ARIMA",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(safrica_test_pred, file = "Results/erica/arima/South Africa_test_pred.jpeg")

# rmse of linear trend + arima
ModelMetrics::rmse(final_test_safrica$value, final_test_safrica$pred) #3241.057






