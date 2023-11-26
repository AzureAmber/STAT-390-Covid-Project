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
train_prophet <- readRDS('data/avg_final_data/final_train_lm.rds')
test_prophet <- readRDS('data/avg_final_data/final_test_lm.rds')

#remove all observations before first COVID cases: 2020-01-19

train_prophet_log <- train_prophet %>% 
  mutate(new_cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0)) %>% 
  filter(date >= as.Date("2020-01-19"))

test_prophet_log <- test_prophet %>% 
  mutate(new_cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0))



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
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = FALSE, 
  seasonality_daily = TRUE,
  changepoint_num = tune(), 
  changepoint_range = tune(),
  prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), 
  prior_scale_holidays = tune()) %>%
  set_engine('prophet')


prophet_log_recipe <- recipe(new_cases_log ~ date + location, data = train_prophet_log) %>%
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
cores.cluster = makePSOCKcluster(10)
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
# jpeg("Models/erica/results/prophet_single/prophet_single_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
# print(prophetsingle_log_autoplot)
# dev.off()
# 
# 
# # 7. Fit Best Model
# # changepoint_num = 100, changepoint_range = 0.6,
# # prior_scale_changepoints = 100, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.001
# 
# prophet_log_model <- prophet_reg(
#    growth = "linear", 
#    season = "additive",
#    seasonality_yearly = FALSE, 
#    seasonality_weekly = FALSE, 
#    seasonality_daily = TRUE,
#    changepoint_num = 100, 
#    changepoint_range = 0.6,
#    prior_scale_changepoints = 100,
#    prior_scale_seasonality = 0.001, 
#    prior_scale_holidays = 0.001) %>%
#    set_engine('prophet')
# 
# prophet_log_recipe <- recipe(new_cases_log ~ date + location, data = train_prophet_log) %>%
#   step_dummy(all_nominal_predictors())
# 
# prophet_log_wflow_tuned <- workflow() %>%
#    add_model(prophet_log_model) %>%
#    add_recipe(prophet_log_recipe)
# 
# prophet_log_fit <- fit(prophet_log_wflow_tuned, data = train_prophet_log)
# 
# final_prophet_train <- train_prophet_log %>%
#   bind_cols(predict(prophet_log_fit, new_data = train_prophet_log)) %>% 
#   mutate(.pred = exp(.pred))
#   
# 
# final_test = test_lm %>%
#   bind_cols(predict(prophet_fit, new_data = test_lm)) %>%
#   mutate(.pred = exp(.pred)) %>%
#   rename(pred = .pred)
# 
# 
# library(ModelMetrics)
# result_train <- final_log_train %>%
#    group_by(location) %>%
#    summarise(value = ModelMetrics::rmse(exp(new_cases), exp(pred))) %>%
#    arrange(location)
#  
# 
# final_log_test <- test_prophet_log %>%
#   bind_cols(predict(prophet_log_fit, new_data = test_prophet_log)) %>%
#   mutate(.pred = exp(.pred)) %>% 
#   rename(pred = .pred)
# 
# result_test <- final_log_test %>%
#   group_by(location) %>%
#   summarise(value = ModelMetrics::rmse(new_cases, pred)) %>%
#   arrange(location)
# 
