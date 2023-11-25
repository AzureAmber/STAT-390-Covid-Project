library(tidyverse)
library(prophet)
library(modeltime)
library(RcppRoll)
library(tidymodels)
library(doParallel)
library(forecast)
library(lubridate)

##### just throw in original data, no need to preprocessing


# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_prophet <- readRDS('data/finalized_data/train_prophet.rds')

train_prophet_log <- train_prophet %>% 
  mutate(new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log)) %>% 
  filter(date > as.Date("2020-01-19"))

test_prophet <- readRDS('data/finalized_data/test_prophet.rds')

test_prophet_log <- test_prophet %>% 
  mutate(new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log))



# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_prophet,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds



# 3. Define model, recipe, and workflow
prophet_log_model <- prophet_reg(
  growth = "linear", season = "additive",
  seasonality_yearly = FALSE, seasonality_weekly = FALSE, seasonality_daily = TRUE,
  changepoint_num = tune(), prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), prior_scale_holidays = tune()) %>%
  set_engine('prophet')

prophet_log_recipe <- recipe(new_cases ~ date + location, data = train_prophet_log) %>%
  step_dummy(all_nominal_predictors())
# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_log_wflow <- workflow() %>%
  add_model(prophet_log_model) %>%
  add_recipe(prophet_log_recipe)


# 4. Setup tuning grid
prophet_log_params <- prophet_log_wflow %>%
  extract_parameter_set_dials()
prophet_log_grid <- grid_regular(prophet_log_params, levels = 3)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(4)
registerDoParallel(cores.cluster)

prophet_log_tuned <- tune_grid(
  prophet_log_wflow,
  resamples = data_folds,
  grid = prophet_log_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)


prophet_log_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
prophetsingle_log_autoplot <- autoplot(prophet_log_tuned, metric = "rmse")
show_best(prophet_log_tuned, metric = "rmse")

# #save autoplot
# jpeg("Models/erica/results/prophet_single_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
# # Print the plot to the device
# print(prophetsingle_autoplot)
# # Close the device
# dev.off()
# 
# # 7. Fit Best Model
# # changepoint_num = 0, prior_scale_changepoints = 0.001,
# # prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001
# 
# 
# prophet_wflow_tuned <- prophet_wflow %>%
#   finalize_workflow(select_best(prophet_tuned, metric = "rmse"))
# 
# prophet_fit <- fit(prophet_wflow_tuned, data = train_lm)
# 
# final_train <- train_lm %>%
#   bind_cols(predict(prophet_fit, new_data = train_lm)) %>%
#   rename(pred = .pred)
# 
# ggplot(final_train) +
#   geom_line(aes(date, new_cases), color = 'red') +
#   geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
#   scale_y_continuous(n.breaks = 15) +
#   facet_wrap(~location, scales = "free_y")
# 
# library(ModelMetrics)
# result_train <- final_train %>%
#   group_by(location) %>%
#   summarise(value = rmse(new_cases, pred)) %>%
#   arrange(location)
# 
# print(result_train)
# 
# 
# 
# final_test <- test_lm %>%
#   bind_cols(predict(prophet_fit, new_data = test_lm)) %>%
#   rename(pred = .pred)
# result_test <- final_test %>%
#   group_by(location) %>%
#   summarise(value = rmse(new_cases, pred)) %>%
#   arrange(location)
# 
# 
# 
# 
# 


