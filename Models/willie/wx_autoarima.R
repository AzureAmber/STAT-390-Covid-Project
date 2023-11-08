library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)

# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm = readRDS('data/finalized_data/final_train_lm.rds')
test_lm = readRDS('data/finalized_data/final_test_lm.rds')

# weekly rolling average
train_lm_fix = train_lm %>%
  filter(location == "United States") %>%
  mutate(new_cases_scaled = roll_sum(new_cases, 7, align = "right", fill = 0) / 7) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup() %>%
  slice(which(row_number() %% 7 == 1))
test_lm_fix = test_lm %>%
  filter(location == "United States") %>%
  mutate(new_cases_scaled = roll_sum(new_cases, 7, align = "right", fill = 0) / 7) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup() %>%
  slice(which(row_number() %% 7 == 1))

ggplot(train_lm_fix, aes(date, new_cases_scaled)) + geom_point()

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds = rolling_origin(
  train_lm_fix,
  initial = 52,
  assess = 4*2,
  skip = 4*4,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow
autoarima_model = arima_reg(
  seasonal_period = 52,
  non_seasonal_ar = tune(), non_seasonal_differences = tune(), non_seasonal_ma = tune(),
  seasonal_ar = 1, seasonal_differences = tune(), seasonal_ma = 1) %>%
  set_engine('auto_arima')

autoarima_recipe = recipe(new_cases_scaled ~ date, data = train_lm_fix)
# View(autoarima_recipe %>% prep() %>% bake(new_data = NULL))

autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

# 4. Setup tuning grid
autoarima_params = autoarima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    non_seasonal_ar = non_seasonal_ar(c(3, 5)),
    non_seasonal_ma = non_seasonal_ma(c(3, 5)),
    seasonal_differences = seasonal_differences(c(0,1))
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
# ar = 3, d = 1, ma = 3, D = 0
autoarima_model = arima_reg(
  seasonal_period = 52,
  non_seasonal_ar = 3, non_seasonal_differences = 1, non_seasonal_ma = 3,
  seasonal_ar = 1, seasonal_differences = 0, seasonal_ma = 1) %>%
  set_engine('auto_arima')
autoarima_recipe = recipe(new_cases_scaled ~ date, data = train_lm_fix)
autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

autoarima_fit = fit(autoarima_wflow, data = train_lm_fix)
final_train = train_lm_fix %>%
  bind_cols(predict(autoarima_fit, new_data = train_lm_fix)) %>%
  rename(pred = .pred)

ggplot(final_train) +
  geom_line(aes(date, new_cases_scaled), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15)

library(ModelMetrics)
results = final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases_scaled, pred)) %>%
  arrange(location)



