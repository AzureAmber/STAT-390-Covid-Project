library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)

# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm = readRDS('data/finalized_data/final_train_lm.rds')
test_lm = readRDS('data/finalized_data/final_test_lm.rds')

train_lm_fix = train_lm %>% filter(location == "United States")

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds = rolling_origin(
  train_lm_fix,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow
prophet_model = prophet_reg(
    growth = "linear", season = "additive",
    seasonality_yearly = FALSE, seasonality_weekly = FALSE, seasonality_daily = TRUE,
    changepoint_num = tune(), prior_scale_changepoints = tune(),
    prior_scale_seasonality = tune(), prior_scale_holidays = tune()) %>%
  set_engine('prophet')

prophet_recipe = recipe(new_cases ~ date, data = train_lm_fix)
# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_wflow = workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

# 4. Setup tuning grid
prophet_params = prophet_wflow %>%
  extract_parameter_set_dials()
prophet_grid = grid_regular(prophet_params, levels = 3)

# 5. Model Tuning
prophet_tuned = tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

prophet_tuned %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
autoplot(prophet_tuned, metric = "rmse")

# 7. Fit Best Model
prophet_model = prophet_reg(
  growth = "linear", season = "additive",
  seasonality_yearly = FALSE, seasonality_weekly = FALSE, seasonality_daily = TRUE,
  changepoint_num = tune(), prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), prior_scale_holidays = tune()) %>%
  set_engine('prophet')
prophet_recipe = recipe(new_cases ~ date, data = train_lm_fix)
prophet_wflow = workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

prophet_fit = fit(prophet_wflow, data = train_lm_fix)
final_train = train_lm_fix %>%
  bind_cols(predict(prophet_fit, new_data = train_lm_fix)) %>%
  rename(pred = .pred)

ggplot(final_train) +
  geom_line(aes(date, new_cases), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15)

library(ModelMetrics)
rmse(final_train$new_cases, final_train$pred)


































