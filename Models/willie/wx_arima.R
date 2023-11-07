library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)

# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm = readRDS('data/finalized_data/final_train_lm.rds')
test_lm = readRDS('data/finalized_data/final_test_lm.rds')

train_lm_fix = train_lm %>%
  group_by(location) %>%
  mutate(new_cases_scaled = (new_cases - mean(new_cases)) / sqrt(var(new_cases))) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()
test_lm_fix = test_lm %>%
  group_by(location) %>%
  mutate(new_cases_scaled = (new_cases - mean(new_cases)) / sqrt(var(new_cases))) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()

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
arima_model = arima_reg(
  seasonal_period = 12,
  non_seasonal_ar = tune(), non_seasonal_differences = tune(), non_seasonal_ma = tune(),
  seasonal_ar = 1, seasonal_differences = tune(), seasonal_ma = 1) %>%
  set_engine('arima')

arima_recipe = recipe(new_cases_scaled ~ date, data = train_lm_fix)
# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

arima_wflow = workflow() %>%
  add_model(arima_model) %>%
  add_recipe(arima_recipe)

# 4. Setup tuning grid
arima_params = arima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    non_seasonal_ar = non_seasonal_ar(c(0, 6)),
    non_seasonal_ma = non_seasonal_ma(c(0, 6)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid = grid_regular(arima_params, levels = 3)

# 5. Model Tuning
arima_tuned = tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

arima_tuned %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
autoplot(arima_tuned, metric = "rmse")

# 7. Fit Best Model
# Increase ar, ma
arima_model = arima_reg(
  seasonal_period = 12,
  non_seasonal_ar = 6, non_seasonal_differences = 0, non_seasonal_ma = 3,
  seasonal_ar = 1, seasonal_differences = 0, seasonal_ma = 1) %>%
  set_engine('arima')
arima_recipe = recipe(new_cases_scaled ~ date, data = train_lm_fix)
arima_wflow = workflow() %>%
  add_model(arima_model) %>%
  add_recipe(arima_recipe)

arima_fit = fit(arima_wflow, data = train_lm_fix)
final_train = train_lm_fix %>%
  bind_cols(predict(arima_fit, new_data = train_lm_fix)) %>%
  rename(pred = .pred)

ggplot(final_train %>% filter(location == "Argentina")) +
  geom_line(aes(date, new_cases_scaled), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15)

library(ModelMetrics)
results = final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases_scaled, pred)) %>%
  arrange(location)



