library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)

# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# 1. Read in data
train_prophet <- readRDS('data/finalized_data/final_train_lm.rds')
test_prophet <- readRDS('data/finalized_data/final_test_lm.rds')

complete_prophet <- train_prophet %>% rbind(test_prophet) %>% 
  filter(date >= as.Date("2020-01-19")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  mutate(
    one_wk_lag = dplyr::lag(new_cases, n = 7, default = 0),
    two_wk_lag = dplyr::lag(new_cases, n = 14, default = 0),
    one_month_lag = dplyr::lag(new_cases, n = 30, default =0)
  )

train_prophet <- complete_prophet %>% 
  filter(date < as.Date("2023-01-01")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()

test_prophet <- complete_prophet %>% 
  filter(date >= as.Date ("2023-01-01")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()


# train_prophet_us <- train_prophet_update %>% 
#   filter(location == "United States")
# test_prophet_us <- test_prophet %>% 
#   filter(location == "United States")

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_prophet,
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
                        data = train_prophet) %>%
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
cores.cluster <- makePSOCKcluster(6)
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
prophet_multiple_best <- show_best(prophet_multi_tuned, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/prophet_multiple/prophet_multiple_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
print(prophet_multiple_autoplot)
dev.off()


# changepoint_num = 50, changepoint_range = 0.9
# prior_scale_changepoints = 0.316, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001

## 7. Fit best model
prophet_multi_model <- prophet_reg(
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = FALSE, 
  seasonality_daily = TRUE,
  changepoint_num = 50, 
  changepoint_range = 0.9,
  prior_scale_changepoints = 0.316,
  prior_scale_seasonality = 0.316, 
  prior_scale_holidays = 0.001) %>%
  set_engine('prophet')

prophet_multi_recipe <- recipe(new_cases ~ .,
                               data = train_prophet) %>%
  step_corr(all_numeric_predictors(), threshold = 0.7) %>% 
  step_dummy(all_nominal_predictors())


prophet_multi_wflow_tuned <- workflow() %>%
  add_model(prophet_multi_model) %>%
  add_recipe(prophet_multi_recipe)

prophet_multi_fit <- fit(prophet_multi_wflow_tuned, data = train_prophet)

final_train <- train_prophet %>% 
  bind_cols(predict(prophet_multi_fit, new_data = train_prophet)) %>% 
  rename(pred = .pred)

library(ModelMetrics)
result_train <- final_train %>%
  group_by(location) %>%
  summarize(rmse_pred_train = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location)

final_test <- test_prophet %>%
  bind_cols(predict(prophet_multi_fit, new_data = test_prophet)) %>%
  rename(pred = .pred)

result_test <- final_test %>%
  group_by(location) %>%
  summarize(rmse_pred_test = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location)

results <- result_train %>% 
  inner_join(result_test, by = "location", suffix = c("rmse_train_pred", "rmse_test_pred"))

write.csv(results, "Results/erica/prophet_single/prophet_single_rmse_results.csv", row.names = FALSE)


## Training Visualization

train_plot <- final_train %>% 
  filter(location == "Italy") %>% 
  ggplot(aes(x=date))+
  geom_line(aes(y = new_cases, color = "Actual New Cases"))+
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")+
  scale_y_continuous(n.breaks = 15)+
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y")+
  theme_minimal()+
  labs(x = "Date",
       y = "New Cases",
       title = "Training: Actual vs. Predicted New Cases in United States in 2023",
       subtitle = "prophet_reg(changepoint_num = 25, changepoint_range = 0.9,
       prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001)",
       caption = "Prophet Univariate",
       color = "")+
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(train_plot, file = "Results/erica/prophet_single/United States_train_pred.jpeg", 
       width=8, height =7, dpi = 300)

#testing visualization
test_plot <- final_test %>% 
  filter(location == "United States") %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = new_cases, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in Argentina in 2023",
       subtitle = "prophet_reg(changepoint_num = 25, changepoint_range = 0.9,
       prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001)",
       caption = "Prophet Univariate",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(test_plot, file = "Results/erica/prophet_single/Argentina_test_pred.jpeg",
       width=8, height =7, dpi = 300)