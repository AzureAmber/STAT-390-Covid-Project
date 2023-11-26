library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)

# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_prophet <- readRDS('data/avg_final_data/final_train_lm.rds')
test_prophet <- readRDS('data/avg_final_data/final_test_lm.rds')


#remove observations before first COVID cases
train_prophet_update <- train_prophet %>% 
  filter(date > as.Date("2020-01-19"))

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_prophet_update,
  initial = 366,
  assess = 30*2,
  skip = 30*6,
  cumulative = FALSE
)
#data_folds

# 3. Define model, recipe, and workflow
prophet_model <- prophet_reg(
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = FALSE, 
  seasonality_daily = TRUE,
  changepoint_num = tune(), 
  changepoint_range = tune(),
  prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), 
  prior_scale_holidays = tune()) %>%
  set_engine('prophet')

prophet_multi_recipe <- recipe(new_cases ~ .,
                        data = train_prophet_update) %>%
  step_corr(all_numeric_predictors(), threshold = 0.7) %>% 
  step_dummy(all_nominal_predictors())

prophet_multi_wflow <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_multi_recipe)


# 4. Setup tuning grid
prophet_multi_params <- prophet_multi_wflow %>%
  extract_parameter_set_dials()

prophet_multi_grid <- grid_regular(prophet_multi_params, levels = 3)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)

prophet_multi_tuned <- tune_grid(
  prophet_multi_wflow,
  resamples = data_folds,
  grid = prophet_multi_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)


prophet_multi_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
prophet_multiple_autoplot <- autoplot(prophet_multi_tuned, metric = "rmse")
show_best(prophet_multi_tuned, metric = "rmse")

# #save autoplot
# jpeg("Models/erica/results/prophet_multiple_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
# # Print the plot to the device
# print(prophet_multiple_autoplot)
# # Close the device
# dev.off()
# 
# # 7. Fit Best Model
# # changepoint_num = 0, prior_scale_changepoints = 0.316,
# # prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001
# prophet_model = prophet_reg(
#   growth = "linear", season = "additive",
#   seasonality_yearly = FALSE, seasonality_weekly = FALSE, seasonality_daily = TRUE,
#   changepoint_num = 0, prior_scale_changepoints = 0.316,
#   prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001) %>%
#   set_engine('prophet')
# prophet_recipe = recipe(new_cases ~ date + location + total_deaths + new_deaths +
#                           gdp_per_capita + month + day_of_week,
#                         data = train_lm) %>%
#   step_dummy(all_nominal_predictors())
# prophet_wflow = workflow() %>%
#   add_model(prophet_model) %>%
#   add_recipe(prophet_recipe)
# 
# prophet_wflow_tuned <- prophet_wflow %>%
#   finalize_workflow(select_best(prophet_tuned, metric = "rmse"))
# 
# prophet_fit <- fit(prophet_wflow, data = train_lm)
# final_train = train_lm %>%
#   bind_cols(predict(prophet_fit, new_data = train_lm)) %>%
#   rename(pred = .pred)
# 
# ggplot(final_train) +
#   geom_line(aes(date, new_cases), color = 'red') +
#   geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
#   scale_y_continuous(n.breaks = 15) +
#   facet_wrap(~location, scales = "free_y")
# 
# library(ModelMetrics)
# result_train <- final_train %>%
#   group_by(location) %>%
#   summarise(value = rmse(new_cases, pred)) %>%
#   arrange(location)
# print(result_train)
# 
# final_test <- test_lm %>%
#   bind_cols(predict(prophet_fit, new_data = test_lm)) %>%
#   rename(pred = .pred)
# result_test <- final_test %>%
#   group_by(location) %>%
#   summarise(value = rmse(new_cases, pred)) %>%
#   arrange(location)
# 
# 
# 
# 
# 
# 
# 
# 
# 
