library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)

# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_lm <- readRDS('data/avg_final_data/final_train_lm.rds')
test_lm <- readRDS('data/avg_final_data/final_test_lm.rds')

# weekly rolling average of log of new cases
complete_lm <- train_lm %>% rbind(test_lm) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(
    cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0),
    value = roll_mean(cases_log, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), cases_log, value)) %>%
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
train_lm_fix = NULL
test_lm_fix = NULL
country_names = unique(train_lm$location)
for (i in country_names) {
  data = train_lm %>% filter(location == i)
  complete_data = complete_lm %>% filter(location == i)
  # find linear model by country
  lm_model = lm(value ~ 0 + time_group + seasonality_group,
                data %>% filter(between(time_group, 13, nrow(data) - 12)))
  x = complete_data %>%
    mutate(
      trend = predict(lm_model, newdata = complete_data),
      slope = as.numeric(coef(lm_model)["time_group"]),
      seasonality_add = trend - slope * time_group,
      err = value - trend) %>%
    mutate_if(is.numeric, round, 5)
  train_lm_fix <<- rbind(train_lm_fix, x %>% filter(date < as.Date("2023-01-01")))
  test_lm_fix <<- rbind(test_lm_fix, x %>% filter(date >= as.Date("2023-01-01")))
}

# first extract countries 

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
autoarima_model <- arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = tune(), non_seasonal_differences = tune(), non_seasonal_ma = tune(),
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('auto_arima')

autoarima_recipe <- recipe(err ~ date, data = train_lm_fix_United.States)
# View(autoarima_recipe %>% prep() %>% bake(new_data = NULL))

autoarima_wflow <- workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

# 5. Setup tuning grid
autoarima_params <- autoarima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    non_seasonal_ar = non_seasonal_ar(c(0,5)),
    non_seasonal_differences = non_seasonal_differences(c(0,2)),
    non_seasonal_ma = non_seasonal_ma(c(0,5)),
    seasonal_ar = seasonal_ar(c(0, 2)),
    seasonal_ma = seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
autoarima_grid <- grid_regular(autoarima_params, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(4)
registerDoParallel(cores.cluster)

autoarima_tuned <- tune_grid(
  autoarima_wflow,
  resamples = data_folds,
  grid = autoarima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)

autoarima_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
autoplot(autoarima_tuned, metric = "rmse")
best_log_combination <- show_best(autoarima_tuned, metric = "rmse")

# 8. Fit Best Model

autoarima_model <- arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 5,
  seasonal_ar = 1, seasonal_differences = 1, seasonal_ma = 0) %>%
  set_engine('auto_arima')
autoarima_recipe <- recipe(err ~ date, data = train_lm_fix_United.States)
autoarima_wflow_tuned <- workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

autoarima_fit <- fit(autoarima_wflow_tuned, data = train_lm_fix_United.States)
final_train <- train_lm_fix_United.States %>%
  bind_cols(pred_err = autoarima_fit$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

# final prediction with linear trend + arima error modelling
ggplot(final_train) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, pred), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Log prediction with linear trend + arima")
ggplot(final_train) +
  geom_line(aes(date, exp(value)), color = 'blue') +
  geom_line(aes(date, exp(pred)), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Prediction with linear trend + arima")

library(ModelMetrics)
# rmse of error prediction
rmse(final_train$err, final_train$pred_err)
# rmse of just linear trend
rmse(final_train$value, final_train$trend)
rmse(exp(final_train$value), exp(final_train$trend))
# rmse of linear trend + arima
rmse(final_train$value, final_train$pred)
ModelMetrics::rmse(exp(final_train$value), exp(final_train$pred))



# Testing set
final_test <- test_lm_fix_United.States %>%
  bind_cols(predict(autoarima_fit, new_data = test_lm_fix_United.States)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

# final prediction with linear trend + arima error modelling
ggplot(final_test) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, pred), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Log prediction with linear trend + arima")
ggplot(final_test) +
  geom_line(aes(date, exp(value)), color = 'blue') +
  geom_line(aes(date, exp(pred)), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Prediction with linear trend + arima")


# rmse of linear trend + arima
rmse(final_test$value, final_test$pred)
ModelMetrics::rmse(exp(final_test$value), exp(final_test$pred))



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
      autoarima_fit <- fit(autoarima_wflow_tuned, get(train_data_name))
      
      # Prepare training data
      final_train <- get(train_data_name) %>%
        bind_cols(pred_err = autoarima_fit$fit$fit$fit$data$.fitted) %>%
        mutate(pred = trend + pred_err) %>%
        mutate_if(is.numeric, round, 5)
      
      # RMSE calculations for training data
      rmse_pred_train <- ModelMetrics::rmse(exp(final_train$value), exp(final_train$pred))
      
      # Prepare testing data
      final_test <- get(test_data_name) %>%
        bind_cols(predict(autoarima_fit, new_data = get(test_data_name))) %>%
        rename(pred_err = .pred) %>%
        mutate(pred = trend + pred_err) %>%
        mutate_if(is.numeric, round, 5)
      
      # RMSE calculations for testing data
      rmse_pred_test <- ModelMetrics::rmse(exp(final_test$value), exp(final_test$pred))
      
      # Check if the final_train and final_test data frames are available
      if (exists("final_train") && exists("final_test")) {
        # Plot for training data
        train_plot <- ggplot(final_train, aes(x=date)) +
          geom_line(aes(y = exp(value), color = "Actual New Cases")) +
          geom_line(aes(y = exp(pred), color = "Predicted New Cases"), linetype = "dashed") +
          scale_y_continuous(n.breaks = 15) + 
          scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
          theme_minimal() + 
          labs(x = "Date", 
               y = "New Cases", 
               title = paste0("Training: Actual vs. Predicted New Cases in ", location),
               subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,5), (P,D,Q) = (1,1,0))",
               caption = "Auto ARIMA",
               color = "") + 
          theme(plot.title = element_text(face = "bold", hjust = 0.5),
                plot.subtitle = element_text(face = "italic", hjust = 0.5),
                legend.position = "bottom",
                panel.grid.minor = element_blank()) +
          scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
        
        # Plot for testing data
        test_plot <- ggplot(final_test, aes(x=date)) +
          geom_line(aes(y = exp(value), color = "Actual New Cases")) +
          geom_line(aes(y = exp(pred), color = "Predicted New Cases"), linetype = "dashed") +
          scale_y_continuous(n.breaks = 15) + 
          scale_x_date(date_breaks = "1 months", date_labels = "%b %y") +
          theme_minimal() + 
          labs(x = "Date", 
               y = "New Cases", 
               title = paste0("Testing: Actual vs. Predicted New Cases in ", location),
               subtitle = "arima_reg(seasonal_period=auto, (p,d,q) = (2,0,5), (P,D,Q) = (1,1,0))",
               caption = "Auto ARIMA",
               color = "") + 
          theme(plot.title = element_text(face = "bold", hjust = 0.5),
                plot.subtitle = element_text(face = "italic", hjust = 0.5),
                legend.position = "bottom",
                panel.grid.minor = element_blank()) +
          scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
        
        ggsave(train_plot, file = paste0("Results/erica/autoarima_log/", location,"_train_pred",  ".jpeg"),
               width=8, height =7, dpi = 300)
        ggsave(test_plot, file = paste0("Results/erica/autoarima_log/", location, "_test_pred", ".jpeg"),
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
write.csv(rmse_results, "Results/erica/autoarima_log/autoarima_rmse_results.csv", row.names = FALSE)









