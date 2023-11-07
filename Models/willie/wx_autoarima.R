library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)

# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm = readRDS('data/finalized_data/final_train_lm.rds')
test_lm = readRDS('data/finalized_data/final_test_lm.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds = rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow
autoarima_model = arima_reg(
  seasonal_period = 12,
  non_seasonal_ar = tune(), non_seasonal_differences = tune(), non_seasonal_ma = tune(),
  seasonal_ar = 1, seasonal_differences = tune(), seasonal_ma = 1) %>%
  set_engine('auto_arima')

autoarima_recipe = recipe(new_cases ~ date, data = train_lm)
# View(autoarima_recipe %>% prep() %>% bake(new_data = NULL))

autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

# 4. Setup tuning grid
autoarima_params = autoarima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    non_seasonal_ar = non_seasonal_ar(c(0, 6)),
    non_seasonal_ma = non_seasonal_ma(c(0, 6)),
    seasonal_differences = seasonal_differences(c(0,2))
  )
autoarima_grid = grid_regular(autoarima_params, levels = 3)

# 5. Model Tuning
autoarima_tuned = tune_grid(
  autoarima_wflow,
  resamples = data_folds,
  grid = autoarima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

autoarima_tuned %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
autoplot(autoarima_tuned, metric = "rmse")

# 7. Fit Best Model
# Increase ar, ma
autoarima_model = arima_reg(
  seasonal_period = 12,
  non_seasonal_ar = 5, non_seasonal_differences = 0, non_seasonal_ma = 5,
  seasonal_ar = 1, seasonal_differences = 0, seasonal_ma = 1) %>%
  set_engine('auto_arima')
autoarima_recipe = recipe(new_cases ~ date, data = train_lm)
autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

autoarima_fit = fit(autoarima_wflow, data = train_lm)
final_train = train_lm %>%
  bind_cols(predict(autoarima_fit, new_data = train_lm)) %>%
  rename(pred = .pred)

ggplot(final_train %>% filter(location == "Argentina")) +
  geom_line(aes(date, new_cases), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15)

library(ModelMetrics)
results = final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)



