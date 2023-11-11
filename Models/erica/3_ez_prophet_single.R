library(tidyverse)
library(prophet)
library(modeltime)
library(doParallel)
library(RcppRoll)
library(dplyr)


##### just throw in original data, no need to preprocessing



# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_lm <- readRDS('data/finalized_data/final_train_lm.rds')
test_lm <- readRDS('data/finalized_data/final_test_lm.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds

# 3. Define model, recipe, and workflow
prophet_model <- prophet_reg(
  growth = "linear", season = "additive",
  seasonality_yearly = FALSE, seasonality_weekly = FALSE, seasonality_daily = TRUE,
  changepoint_num = tune(), prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), prior_scale_holidays = tune()) %>%
  set_engine('prophet')

prophet_recipe <- recipe(new_cases ~ date + location, data = train_lm) %>%
  step_dummy(all_nominal_predictors())
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
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

prophet_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
autoplot(prophet_tuned, metric = "rmse")
show_best(prophet_tune, metric = "rmse")

# 7. Fit Best Model
# changepoint_num = 0, prior_scale_changepoints = 0.001,
# prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001


prophet_wflow_tuned <- prophet_wflow %>%
  finalize_workflow(select_best(prophet_tuned, metric = "rmse"))

prophet_fit <- fit(prophet_wflow_tuned, data = train_lm)

final_train <- train_lm %>%
  bind_cols(predict(prophet_fit, new_data = train_lm)) %>%
  rename(pred = .pred)

ggplot(final_train) +
  geom_line(aes(date, new_cases), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  facet_wrap(~location, scales = "free_y")

library(ModelMetrics)
result <- final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)

print(result)



final_test <- test_lm %>%
  bind_cols(predict(prophet_fit, new_data = test_lm)) %>%
  rename(pred = .pred)
result_test <- final_test %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)





