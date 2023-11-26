library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)

# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
final_train_lm = readRDS('data/avg_final_data/final_train_lm.rds')
final_test_lm = readRDS('data/avg_final_data/final_test_lm.rds')

# weekly rolling sum of log of new cases
# Remove observations before first appearance of COVID: 2020-01-04
complete_lm = final_train_lm %>% rbind(final_test_lm) %>%
  filter(date >= as.Date("2020-01-04")) %>%
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

# plots of some countries
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
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, trend), color = 'red')
# plot of residual errors
ggplot(train_lm_fix_init %>% filter(location == "United States"), aes(date, err)) + geom_line()

# 3 ARIMA model for US data
# Find best arima parameters to model the error after removing trend
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

# 3.1. Define model, recipe, and workflow
autoarima_model = arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = tune(), non_seasonal_differences = tune(), non_seasonal_ma = tune(),
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('auto_arima')

autoarima_recipe = recipe(err ~ date, data = train_lm_fix)
# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

autoarima_wflow = workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

# 3.2. Setup tuning grid
autoarima_params = autoarima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    non_seasonal_differences = non_seasonal_differences(c(0,2)),
    non_seasonal_ar = non_seasonal_ar(c(0, 4)),
    non_seasonal_ma = non_seasonal_ma(c(0, 4)),
    seasonal_differences = seasonal_differences(c(0,2)),
    seasonal_ar = seasonal_ar(c(0, 2)),
    seasonal_ma = seasonal_ma(c(0, 2))
  )
autoarima_grid = grid_regular(autoarima_params, levels = 3)

# 3.3. Model Tuning
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
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)

autoarima_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 3.4. Results
autoplot(autoarima_tuned, metric = "rmse")










# 4. Fit Best Model
train_lm_fix = train_lm_fix_init %>% filter(location == "Germany")
test_lm_fix = test_lm_fix_init %>% filter(location == "Germany")

autoarima_model = arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 0,
  seasonal_ar = 0, seasonal_differences = 0, seasonal_ma = 0) %>%
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


library(ModelMetrics)
# rmse of just linear trend
rmse(exp(final_train$value), exp(final_train$trend))
# rmse of linear trend + arima
rmse(exp(final_train$value), exp(final_train$pred))



# Testing set
final_test = test_lm_fix %>%
  bind_cols(predict(autoarima_fit, new_data = test_lm_fix)) %>%
  rename(pred_err = .pred) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

# rmse of just linear trend
rmse(exp(final_test$value), exp(final_test$trend))
# rmse of linear trend + arima
rmse(exp(final_test$value), exp(final_test$pred))



# plot
x = final_train %>% 
  select(date, value, pred) %>%
  pivot_longer(cols = c("value", "pred"), names_to = "type", values_to = "value") %>%
  mutate(
    type = ifelse(type == 'value', 'New Cases', 'Predicted New Cases'),
    type = factor(type, levels = c('New Cases', 'Predicted New Cases'))
  )



ggplot(x, aes(date, exp(value))) +
  geom_line(aes(color = type, linetype = type)) +
  scale_y_continuous(n.breaks = 10) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  scale_color_manual(values = c("red", "blue")) +
  labs(
    title = "Log Training: Actual vs Predicted New Cases in Germany",
    x = "Date", y = "New Cases") +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 20),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#808080"))





y = final_test %>% 
  select(date, value, pred) %>%
  pivot_longer(cols = c("value", "pred"), names_to = "type", values_to = "value") %>%
  mutate(
    type = ifelse(type == 'value', 'New Cases', 'Predicted New Cases'),
    type = factor(type, levels = c('New Cases', 'Predicted New Cases'))
  )



ggplot(y, aes(date, exp(value))) +
  geom_line(aes(color = type, linetype = type)) +
  scale_y_continuous(n.breaks = 10) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  scale_color_manual(values = c("red", "blue")) +
  labs(
    title = "Log Testing: Actual vs Predicted New Cases in Germany",
    x = "Date", y = "New Cases") +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 20),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#808080"))





