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



#remove observations before first COVID cases
train_prophet_log <- train_prophet %>% 
  filter(date >= as.Date("2020-01-19")) %>% 
  mutate(new_cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0))

test_prophet_log <- test_prophet %>% 
  mutate(new_cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0))

#weekly rolling average of new cases

complete_lm_log <- train_prophet_log %>% rbind(test_prophet_log) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>%
  mutate(value = roll_mean(new_cases_log, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), new_cases_log, value)) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(which(row_number() %% 7 == 0)) %>%
  mutate(
    time_group = row_number(),
    seasonality_group = row_number() %% 53) %>%
  ungroup() %>%
  mutate(seasonality_group = as.factor(seasonality_group))

train_prophet_log <- complete_lm_log %>% filter(date < as.Date("2023-01-01")) %>%
  group_by(date) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()
test_prophet_log <- complete_lm_log %>% filter(date >= as.Date("2023-01-01")) %>%
  group_by(date) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()


# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_prophet_log,
  initial = 53*23,
  assess = 4*2*23,
  skip = 4*4*23,
  cumulative = FALSE
)
#data_folds


# 3. Define model, recipe, and workflow
prophet_model <- prophet_reg(
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = TRUE, 
  seasonality_daily = FALSE,
  changepoint_num = tune(), 
  changepoint_range = tune(),
  prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), 
  prior_scale_holidays = tune()) %>%
  set_engine('prophet')

prophet_recipe <- recipe(value ~ ., data = train_prophet_log) %>%
  step_rm(day_of_week, continent) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.7) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())
# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_wflow <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)


# 4. Setup tuning grid
prophet_params <- prophet_wflow %>%
  extract_parameter_set_dials()
prophet_grid <- grid_regular(prophet_params, levels = 3)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(5)
registerDoParallel(cores.cluster)

prophet_tuned <- tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)


prophet_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
autoplot(prophet_tuned, metric = "rmse")
show_best(prophet_tuned, metric = "rmse")
