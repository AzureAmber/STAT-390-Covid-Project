library(tidyverse)
library(prophet)
library(modeltime)
library(RcppRoll)
library(tidymodels)
library(doParallel)
library(forecast)
library(lubridate)

##### just throw in original data, no need to preprocessing


# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_prophet <- readRDS('data/avg_final_data/final_train_lm.rds')
test_prophet <- readRDS('data/avg_final_data/final_test_lm.rds')



#remove observations before first COVID cases
train_prophet_log <- train_prophet %>% 
  filter(date >= as.Date("2020-01-19")) %>% 
  mutate(new_cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0))

test_prophet_log <- test_prophet %>% 
  mutate(new_cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0))

#weekly rolling average of new cases

complete_lm_log <- train_prophet_log %>% rbind(test_prophet_log) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>%
  mutate(value = roll_mean(new_cases_log, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), new_cases_log, value)) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(which(row_number() %% 7 == 0)) %>%
  mutate(
    time_group = row_number(),
    seasonality_group = row_number() %% 53) %>%
  ungroup() %>%
  mutate(seasonality_group = as.factor(seasonality_group))

train_prophet_log <- complete_lm_log %>% filter(date < as.Date("2023-01-01")) %>%
  group_by(date) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()
test_prophet_log <- complete_lm_log %>% filter(date >= as.Date("2023-01-01")) %>%
  group_by(date) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()


# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_prophet_log,
  initial = 53*23,
  assess = 4*2*23,
  skip = 4*4*23,
  cumulative = FALSE
)
#data_folds


# 3. Define model, recipe, and workflow
prophet_model <- prophet_reg(
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = TRUE, 
  seasonality_daily = FALSE,
  changepoint_num = tune(), 
  changepoint_range = tune(),
  prior_scale_changepoints = tune(),
  prior_scale_seasonality = tune(), 
  prior_scale_holidays = tune()) %>%
  set_engine('prophet')

prophet_recipe <- recipe(value ~ ., data = train_prophet_log) %>%
  step_rm(day_of_week, continent) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.7) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())
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
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)


prophet_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
autoplot(prophet_tuned, metric = "rmse")
show_best(prophet_tuned, metric = "rmse")

# 7. Fitting Model

# changepoint_num = 50, changepoint_range = 0.6
# prior_scale_changepoints = 100, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001

prophet_multi_model <- prophet_reg(
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = TRUE, 
  seasonality_daily = FALSE,
  changepoint_num = 50, 
  changepoint_range = 0.6,
  prior_scale_changepoints = 100,
  prior_scale_seasonality = 0.316, 
  prior_scale_holidays = 0.001) %>%
  set_engine('prophet')

prophet_multi_recipe <- recipe(value ~ .,
                               data = train_prophet_log) %>%
  step_rm(day_of_week, continent, G20, G24) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.7) %>% 
  step_dummy(all_nominal_predictors())


prophet_multi_wflow_tuned <- workflow() %>%
  add_model(prophet_multi_model) %>%
  add_recipe(prophet_multi_recipe)

prophet_multi_fit <- fit(prophet_multi_wflow_tuned, data = train_prophet_log)

final_multi_train <- predict(prophet_multi_fit, new_data = train_prophet_log) %>% 
  bind_cols(train_prophet_log) %>% 
  mutate(.pred = exp(.pred)) %>%
  rename(pred = .pred)

final_multi_test <- test_prophet_log %>%
  bind_cols(predict(prophet_multi_fit, new_data = test_prophet_log)) %>%
  mutate(.pred = exp(.pred)) %>%
  rename(pred = .pred)

library(ModelMetrics)
result_multi_train <- final_multi_train %>%
  group_by(location) %>%
  summarize(rmse_pred_train = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location)

result_multi_test <- final_multi_test %>%
  group_by(location) %>%
  summarize(rmse_pred_test = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location)

results_multi <- result_multi_train %>% 
  inner_join(result_multi_test, by = "location", suffix = c("rmse_train_pred", "rmse_test_pred"))

write.csv(results_multi, "Results/erica/prophet_multi_log/prophet_multi_log_rmse_results.csv", row.names = FALSE)

## Training + Testing Visualization

countries <- unique(final_multi_train$location)

for (loc in countries){
  train_title <- paste0("Training: Actual vs Predicted New Cases in ", loc, " in 2023")
  train_file <- paste0("Results/erica/prophet_multi_log/", loc, "_train_pred.jpeg")
  
  test_title <- paste0("Testing: Actual vs Predicted New Cases in ", loc, " in 2023")
  test_file <- paste0("Results/erica/prophet_multi_log/", loc, "_test_pred.jpeg")
  
  train_plot <- final_multi_train %>% 
    filter(location == loc) %>%
    ggplot(aes(x=date))+
    geom_line(aes(y = new_cases, color = "Actual New Cases"))+
    geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")+
    scale_y_continuous(n.breaks = 15)+
    scale_x_date(date_breaks = "3 months", date_labels = "%b %y")+
    theme_minimal()+
    labs(x = "Date",
         y = "New Cases",
         title = train_title,
         subtitle = "prophet_reg(changepoint_num = 50, changepoint_range = 0.6,
       prior_scale_changepoints = 100, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001)",
         caption = "Prophet Multivariate Log",
         color = "")+
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  ggsave(train_plot, file = train_file, width=8, height =7, dpi = 300)
  
  test_plot <- final_multi_test %>% 
    filter(location == loc) %>% 
    ggplot(aes(x=date)) +
    geom_line(aes(y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = test_title,
         subtitle = "prophet_reg(changepoint_num = 50, changepoint_range = 0.6,
       prior_scale_changepoints = 100, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001",
         caption = "Prophet Multivariate Log",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  ggsave(test_plot, file = test_file, width=8, height =7, dpi = 300)
  
}




