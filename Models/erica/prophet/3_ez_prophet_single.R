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
train_prophet_update <- train_prophet %>% 
  filter(date > as.Date("2020-01-19"))


# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_prophet_update,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
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

prophet_recipe <- recipe(new_cases ~ date + location, data = train_prophet_update) %>%
  step_dummy(all_nominal_predictors())
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
cores.cluster = makePSOCKcluster(4)
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
prophetsingle_autoplot <- autoplot(prophet_tuned, metric = "rmse")
best_prophet_single <- show_best(prophet_tuned, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/prophet_single/prophet_single_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
print(prophetsingle_autoplot)
dev.off()

# 7. Fit Best Model


# # A tibble: 5 × 11
# changepoint_num changepoint_range prior_scale_changepoints prior_scale_seasonality prior_scale_holidays .metric
# <int>             <dbl>                    <dbl>                   <dbl>                <dbl> <chr>  
#   1              50              0.9                     0.001                   0.001                0.001 rmse   
# 2              50              0.9                     0.001                   0.001                0.316 rmse   
# 3              50              0.9                     0.001                   0.001              100     rmse   
# 4              50              0.75                    0.001                   0.001                0.001 rmse   
# 5              50              0.75                    0.001                   0.001                0.316 rmse  


# changepoint_num = 50, changepoint_range = 0.9
# prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.001
# 
# prophet_model <- prophet_reg(
#   growth = "linear", 
#   season = "additive",
#   seasonality_yearly = FALSE, 
#   seasonality_weekly = FALSE, 
#   seasonality_daily = TRUE,
#   changepoint_num = 0, 
#   changepoint_range = 0.6,
#   prior_scale_changepoints = 100,
#   prior_scale_seasonality = 0.001, 
#   prior_scale_holidays = 0.316) %>%
#   set_engine('prophet')
# 
# 
# prophet_recipe <- recipe(new_cases ~ date + location, data = train_prophet_update) %>%
#   step_dummy(all_nominal_predictors())
# 
# 
# prophet_wflow_tuned <- workflow() %>%
#   add_model(prophet_model) %>%
#   add_recipe(prophet_recipe)
# 
# prophet_fit <- fit(prophet_wflow_tuned, data = train_prophet_update)
# 
# final_train <- predict(prophet_fit, new_data = train_prophet_update) %>% 
#   bind_cols(train_prophet_update) %>% 
#   rename(pred = .pred)
# 
# 
# library(ModelMetrics)
# 
# result_train <- final_train %>%
#   group_by(location) %>%
#   summarize(rmse_pred_train = ModelMetrics::rmse(new_cases, pred)) %>%
#   arrange(location)
# 
# 
# final_test <- test_prophet %>%
#   bind_cols(predict(prophet_fit, new_data = test_prophet)) %>%
#   rename(pred = .pred)
# 
# result_test <- final_test %>%
#   group_by(location) %>%
#   summarize(rmse_pred_test = ModelMetrics::rmse(new_cases, pred)) %>%
#   arrange(location)
# 
# results <- result_train %>% 
#   inner_join(result_test, by = "location", suffix = c("rmse_train_pred", "rmse_test_pred"))
# 
# write.csv(results, "Results/erica/prophet_single/prophet_single_rmse_results.csv", row.names = FALSE)
# 
# ## Training Visualization
# 
# train_plot <- final_train %>% 
#   filter(location == "United States") %>% 
#   ggplot(aes(x=date))+
#   geom_line(aes(y = new_cases, color = "Actual New Cases"))+
#   geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")+
#   scale_y_continuous(n.breaks = 15)+
#   scale_x_date(date_breaks = "3 months", date_labels = "%b %y")+
#   theme_minimal()+
#   labs(x = "Date",
#        y = "New Cases",
#        title = paste0("Training: Actual vs. Predicted New Cases in United States in 2023"),
#        subtitle = "prophet_reg(changepoint_num = 0, changepoint_range = 0.6,
#          prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.316)",
#        caption = "Prophet Univariate",
#        color = "")+
#   theme(plot.title = element_text(face = "bold", hjust = 0.5),
#         plot.subtitle = element_text(face = "italic", hjust = 0.5),
#         legend.position = "bottom",
#         panel.grid.minor = element_blank()) +
#   scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
# 
# ggsave(train_plot, file = "Results/erica/prophet_single/United States_train_pred.jpeg",
#        width=8, height =7, dpi = 300)
# 
# #testing visualization
# test_plot <- final_test %>% 
#   filter(location == "Argentina") %>% 
#   ggplot(aes(x=date)) +
#   geom_line(aes(y = new_cases, color = "Actual New Cases")) +
#   geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
#   scale_y_continuous(n.breaks = 15) + 
#   scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
#   theme_minimal() + 
#   labs(x = "Date", 
#        y = "New Cases", 
#        title = paste0("Testing: Actual vs. Predicted New Cases in Argentina in 2023"),
#        subtitle = "prophet_reg(changepoint_num = 0, changepoint_range = 0.6,
#          prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.316)",
#        caption = "Prophet Univariate",
#        color = "") + 
#   theme(plot.title = element_text(face = "bold", hjust = 0.5),
#         plot.subtitle = element_text(face = "italic", hjust = 0.5),
#         legend.position = "bottom",
#         panel.grid.minor = element_blank()) +
#   scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
# 
# ggsave(test_plot, file = "Results/erica/prophet_single/Argentina_test_pred.jpeg",
#        width=8, height =7, dpi = 300)
# 
# 
# 
# 
