library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)

# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
# Remove observations before first appearance of COVID: 2020-01-04
train_lm = readRDS('data/avg_final_data/final_train_lm.rds') %>% filter(date >= as.Date("2020-01-04"))
test_lm = readRDS('data/avg_final_data/final_test_lm.rds')

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
prophet_model = prophet_reg(
  growth = "linear", season = "additive",
  seasonality_yearly = FALSE, seasonality_weekly = FALSE, seasonality_daily = TRUE,
  changepoint_num = tune(), changepoint_range = tune(), prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), prior_scale_holidays = tune()) %>%
  set_engine('prophet')

# apply a log transformation to response = new_cases
prophet_recipe = recipe(new_cases ~ ., data = train_lm) %>%
  step_mutate(new_cases = ifelse(is.finite(log(new_cases)), log(new_cases), 0)) %>%
  step_corr(all_numeric_predictors(), threshold = 0.7) %>%
  step_dummy(all_nominal_predictors())
# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_wflow = workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

# 4. Setup tuning grid
prophet_params = prophet_wflow %>%
  extract_parameter_set_dials()
prophet_grid = grid_regular(prophet_params, levels = 3)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)

prophet_tuned = tune_grid(
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

# 7. Fit Best Model
prophet_model = prophet_reg(
  growth = "linear", season = "additive",
  seasonality_yearly = FALSE, seasonality_weekly = FALSE, seasonality_daily = TRUE,
  changepoint_num = 50, changepoint_range = tune(), prior_scale_changepoints = 100,
  prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001) %>%
  set_engine('prophet')
prophet_recipe = recipe(new_cases ~ ., data = train_lm) %>%
  step_mutate(new_cases = ifelse(is.finite(log(new_cases)), log(new_cases), 0)) %>%
  step_corr(all_numeric_predictors(), threshold = 0.7) %>%
  step_dummy(all_nominal_predictors())
prophet_wflow = workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

prophet_fit = fit(prophet_wflow, data = train_lm)
final_train = train_lm %>%
  bind_cols(predict(prophet_fit, new_data = train_lm)) %>%
  rename(pred = .pred)
final_test = test_lm %>%
  bind_cols(predict(prophet_fit, new_data = test_lm)) %>%
  rename(pred = .pred)


# rmse
library(ModelMetrics)
result = final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)
result_test = final_test %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)



# plot
ggplot(final_train) +
  geom_line(aes(date, new_cases), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  facet_wrap(~location, scales = "free_y")



# plots
x = final_train %>%
  filter(location == "Germany") %>%
  select(date, new_cases, pred) %>%
  pivot_longer(cols = c("new_cases", "pred"), names_to = "type", values_to = "value") %>%
  mutate(
    type = ifelse(type == 'new_cases', 'New Cases', 'Predicted New Cases'),
    type = factor(type, levels = c('New Cases', 'Predicted New Cases'))
  )



ggplot(x, aes(date, value)) +
  geom_line(aes(color = type, linetype = type)) +
  scale_y_continuous(n.breaks = 10) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  scale_color_manual(values = c("red", "blue")) +
  labs(
    title = "Training: Actual vs Predicted New Cases in United States",
    subtitle = "prophet_reg(changepoint_num = 0, prior_scale_changepoints = 0.001, \n prior_scale_seasonality = 0.216, prior_scale_holidays = 0.001)",
    x = "Date", y = "New Cases") +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 20),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#808080"))





y = final_test %>% 
  filter(location == "Germany") %>%
  select(date, new_cases, pred) %>%
  pivot_longer(cols = c("new_cases", "pred"), names_to = "type", values_to = "value") %>%
  mutate(
    type = ifelse(type == 'new_cases', 'New Cases', 'Predicted New Cases'),
    type = factor(type, levels = c('New Cases', 'Predicted New Cases'))
  )



ggplot(y, aes(date, value)) +
  geom_line(aes(color = type, linetype = type)) +
  scale_y_continuous(n.breaks = 10) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  scale_color_manual(values = c("red", "blue")) +
  labs(
    title = "Testing: Actual vs Predicted New Cases in United States",
    subtitle = "prophet_reg(changepoint_num = 0, prior_scale_changepoints = 0.001, \n prior_scale_seasonality = 0.216, prior_scale_holidays = 0.001)",
    x = "Date", y = "New Cases") +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 20),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#808080"))




