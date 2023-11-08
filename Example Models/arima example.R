library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(lubridate)
library(doParallel)
library(RcppRoll)

data = bike_sharing_daily %>%
  select(dteday, cnt) %>%
  set_names(c("date", "value")) %>%
  mutate(
    value_avg = roll_mean(value, n = 7, fill = NA, align = "right"),
    value_avg = ifelse(is.na(value_avg), value, value_avg)) %>%
  slice(which(row_number() %% 7 == 1))
ggplot(data, aes(date, value_avg)) +
  geom_line()

# determine model trend
M = factor(rep(1:53, length.out = nrow(data)))
lm_model = lm(value_avg ~ 0 + time(date) + M, data)
lm_model
x = data %>% mutate(trend = predict(lm_model, newdata = data)) %>%
  mutate(err = value_avg - trend) %>%
  mutate_if(is.numeric, round, 5)
# plot of original data and trend
ggplot(x) +
  geom_line(aes(date, value_avg), color = 'blue') +
  geom_line(aes(date, trend), color = 'red')
# plot of residual errors
ggplot(x, aes(date, err)) + geom_line()



# determine arima parameters to model err
splits = x %>% time_series_split(assess = "3 months", cumulative = TRUE)
data_train = training(splits)
data_test = testing(splits)

data_folds = rolling_origin(
  data_train,
  initial = 53,
  assess = 4*2,
  skip = 4*2,
  cumulative = TRUE
)
data_folds

# model tuning
autoarima_model = arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = tune(), non_seasonal_differences = tune(), non_seasonal_ma = tune(),
    seasonal_ar = 1, seasonal_differences = 0, seasonal_ma = 1) %>%
  set_engine('auto_arima')
autoarima_recipe = recipe(err ~ date, data = data_train)
autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

autoarima_params = autoarima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    non_seasonal_ar = non_seasonal_ar(c(1, 5)),
    non_seasonal_ma = non_seasonal_ma(c(1, 5)),
    non_seasonal_differences = non_seasonal_differences(c(0,1))
  )
autoarima_grid = grid_regular(autoarima_params, levels = 3)


cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)

autoarima_tuned = tune_grid(
  autoarima_wflow,
  resamples = data_folds,
  grid = autoarima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)



autoarima_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)
autoplot(autoarima_tuned, metric = "rmse")



# fit arima model
autoarima_model = arima_reg(
  seasonal_period = 48,
  non_seasonal_ar = 1, non_seasonal_differences = 0, non_seasonal_ma = 1,
  seasonal_ar = 1, seasonal_differences = 0, seasonal_ma = 1) %>%
  set_engine('auto_arima')
autoarima_recipe = recipe(err ~ date, data = data_train)
autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

autoarima_fit = fit(autoarima_wflow, data = data_train)
autoarima_fit
final_train = data_train %>%
  bind_cols(pred_err = autoarima_fit$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

# error model
ggplot(final_train) +
  geom_line(aes(date, err), color = 'blue') +
  geom_line(aes(date, pred_err), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15)
# prediction model
ggplot(final_train) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, pred), color = 'red', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15)

library(ModelMetrics)
rmse(final_train$err, final_train$pred_err)
rmse(final_train$value_avg, final_train$pred)


# library(astsa)
# sarima.for(data_train$err, n.ahead = 50, p = 1, d = 0, q = 1, P = 1, D = 1, Q = 1, S = 40)
# plot.new()
# frame()

