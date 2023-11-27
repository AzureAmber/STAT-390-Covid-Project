library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)

# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_prophet <- readRDS('data/avg_final_data/final_train_lm.rds')
test_prophet <- readRDS('data/avg_final_data/final_test_lm.rds')

complete_multi_prophet <- train_prophet %>% rbind(test_prophet) %>% 
  filter(date >= as.Date("2020-01-19")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  mutate(
    one_wk_lag = dplyr::lag(new_cases, n = 7, default = 0),
    two_wk_lag = dplyr::lag(new_cases, n = 14, default = 0),
    one_month_lag = dplyr::lag(new_cases, n = 30, default =0)
  ) %>% 
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), new_cases, value)) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(which(row_number() %% 7 == 0)) %>%
  mutate(
    time_group = row_number(),
    seasonality_group = row_number() %% 53) %>%
  ungroup() %>%
  mutate(seasonality_group = as.factor(seasonality_group))
  

train_multi_prophet <- complete_multi_prophet %>% 
  filter(date < as.Date("2023-01-01")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()

test_multi_prophet <- complete_multi_prophet %>% 
  filter(date >= as.Date ("2023-01-01")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()


# train_prophet_us <- train_prophet_update %>% 
#   filter(location == "United States")
# test_prophet_us <- test_prophet %>% 
#   filter(location == "United States")

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds_multi <- rolling_origin(
  train_multi_prophet,
  initial = 53*23,
  assess = 4*2*23,
  skip = 4*4*23,
  cumulative = FALSE
)
#data_folds

# 3. Define model, recipe, and workflow
prophet_multi_model <- prophet_reg(
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

prophet_multi_recipe <- recipe(value ~ .,
                        data = train_multi_prophet) %>%
  step_rm(day_of_week, continent, G20, G24) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.7) %>% 
  step_dummy(all_nominal_predictors())


prophet_multi_wflow <- workflow() %>%
  add_model(prophet_multi_model) %>%
  add_recipe(prophet_multi_recipe)


# 4. Setup tuning grid
prophet_multi_params <- prophet_multi_wflow %>%
  extract_parameter_set_dials()

prophet_multi_grid <- grid_regular(prophet_multi_params, levels = 3)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)

prophet_multi_tuned <- tune_grid(
  prophet_multi_wflow,
  resamples = data_folds_multi,
  grid = prophet_multi_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)


prophet_multi_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
prophet_multiple_autoplot <- autoplot(prophet_multi_tuned, metric = "rmse")
prophet_multiple_best <- show_best(prophet_multi_tuned, metric = "rmse")


