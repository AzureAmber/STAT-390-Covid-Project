library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)

# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_lm = readRDS('data/finalized_data/final_train_lm.rds')
test_lm = readRDS('data/finalized_data/final_test_lm.rds')

# weekly rolling average of log of new cases
complete_lm = train_lm %>% rbind(test_lm) %>%
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
train_lm = complete_lm %>% filter(date < as.Date("2023-01-01")) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()
test_lm = complete_lm %>% filter(date >= as.Date("2023-01-01")) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()


ggplot(train_lm %>% filter(location == "United States"), aes(date, value)) +
  geom_point()
ggplot(train_lm %>% filter(location == "Germany"), aes(date, value)) +
  geom_point()
ggplot(train_lm %>% filter(location == "South Korea"), aes(date, value)) +
  geom_point()

# 2. Find model trend by country
train_lm_fix_init = NULL
test_lm_fix_init = NULL
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
  train_lm_fix_init <<- rbind(train_lm_fix_init, x %>% filter(date < as.Date("2023-01-01")))
  test_lm_fix_init <<- rbind(test_lm_fix_init, x %>% filter(date >= as.Date("2023-01-01")))
}
# plot of original data and trend
ggplot(train_lm_fix_init %>% filter(location == "United States")) +
  geom_line(aes(date, exp(value)), color = 'blue') +
  geom_line(aes(date, exp(trend)), color = 'red') +
  labs(title = "Original rolling average vs prediction")
ggplot(train_lm_fix_init %>% filter(location == "United States")) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, trend), color = 'red') +
  labs(title = "Log transformed rolling average vs log prediction")
# plot of residual errors
ggplot(train_lm_fix_init %>% filter(location == "United States"), aes(date, err)) + geom_line()





# ARIMA Model tuning for errors
# 3. Create validation sets for every year train + 2 month test with 4-month increments
train_lm_fix = train_lm_fix_init %>% filter(location == "United States")
test_lm_fix = test_lm_fix_init %>% filter(location == "United States")

data_folds = rolling_origin(
  train_lm_fix,
  initial = 53,
  assess = 4*2,
  skip = 4*4,
  cumulative = FALSE
)
data_folds

# 4. Define model, recipe, and workflow
autoarima_model = arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = tune(), non_seasonal_differences = tune(), non_seasonal_ma = tune(),
  seasonal_ar = 1, seasonal_differences = tune(), seasonal_ma = 1) %>%
  set_engine('auto_arima')

autoarima_recipe = recipe(err ~ date, data = train_lm_fix)
# View(autoarima_recipe %>% prep() %>% bake(new_data = NULL))

autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

# 5. Setup tuning grid
autoarima_params = autoarima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    non_seasonal_differences = non_seasonal_differences(c(0,2)),
    non_seasonal_ar = non_seasonal_ar(c(3, 5)),
    non_seasonal_ma = non_seasonal_ma(c(3, 5)),
    seasonal_differences = seasonal_differences(c(0,2))
  )
autoarima_grid = grid_regular(autoarima_params, levels = 3)

# 6. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)

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
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 7. Results
autoplot(autoarima_tuned, metric = "rmse")

# 8. Fit Best Model
# p = 3, d = 1, q = 3, D = 0
autoarima_model = arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = 3, non_seasonal_differences = 1, non_seasonal_ma = 3,
  seasonal_ar = 1, seasonal_differences = 0, seasonal_ma = 1) %>%
  set_engine('auto_arima')
autoarima_recipe = recipe(err ~ date, data = train_lm_fix)
autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

autoarima_fit = fit(autoarima_wflow, data = train_lm_fix)
final_train = train_lm_fix %>%
  bind_cols(pred_err = autoarima_fit$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

# error model
ggplot(final_train) +
  geom_line(aes(date, err), color = 'blue') +
  geom_line(aes(date, pred_err), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Log Error vs Log Error Prediction")
# prediction models
# initial prediction with just linear trend
ggplot(final_train) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, trend), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Log prediction with only linear trend")
ggplot(final_train) +
  geom_line(aes(date, exp(value)), color = 'blue') +
  geom_line(aes(date, exp(trend)), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Prediction with only linear trend")
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
rmse(exp(final_train$value), exp(final_train$pred))



# Testing set
final_test = test_lm_fix %>%
  bind_cols(predict(autoarima_fit, new_data = test_lm_fix)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)
# initial prediction with just linear trend
ggplot(final_test) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, trend), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Log prediction with only linear trend")
ggplot(final_test) +
  geom_line(aes(date, exp(value)), color = 'blue') +
  geom_line(aes(date, exp(trend)), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  labs(title = "Prediction with only linear trend")
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

# rmse of just linear trend
rmse(final_test$value, final_test$trend)
rmse(exp(final_test$value), exp(final_test$trend))
# rmse of linear trend + arima
rmse(final_test$value, final_test$pred)
rmse(exp(final_test$value), exp(final_test$pred))








