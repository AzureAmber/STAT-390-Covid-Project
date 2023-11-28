library(tidyverse)
library(prophet)
library(modeltime)
library(RcppRoll)
library(tidymodels)
library(doParallel)
library(forecast)
library(lubridate)
library(ModelMetrics)

##### just throw in original data, no need to preprocessing


# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0


# 1. Read in data
train_prophet <- readRDS('data/avg_final_data/final_train_lm.rds')
test_prophet <- readRDS('data/avg_final_data/final_test_lm.rds')

#weekly rolling average of new cases
#remove observations before first COVID
complete_uni_prophet <- train_prophet %>% rbind(test_prophet) %>% 
  filter(date >= as.Date("2020-01-19")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), new_cases, value)) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(which(row_number() %% 7 == 0)) %>%
  mutate(
    time_group = row_number(),
    seasonality_group = row_number() %% 53) %>%
  ungroup() %>%
  mutate(seasonality_group = as.factor(seasonality_group))

train_uni_prophet <- complete_uni_prophet %>% 
  filter(date < as.Date("2023-01-01")) %>% 
  group_by(date) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()

test_uni_prophet <- complete_uni_prophet %>% 
  filter(date >= as.Date ("2023-01-01")) %>% 
  group_by(date) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()

# train_prophet_sri <- train_prophet_update %>% 
#   filter(location == "Sri Lanka")
# test_prophet_sri <- test_prophet %>% 
#   filter(location == "Sri Lanka")


# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds_uni <- rolling_origin(
  train_uni_prophet,
  initial = 52*23,
  assess = 4*2*23,
  skip = 4*4*23,
  cumulative = FALSE
)
#data_folds



# 3. Define model, recipe, and workflow
prophet_uni_model <- prophet_reg(
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

prophet_uni_recipe <- recipe(value ~ date + location, data = train_uni_prophet) %>%
  step_dummy(all_nominal_predictors())
# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_uni_wflow <- workflow() %>%
  add_model(prophet_uni_model) %>%
  add_recipe(prophet_uni_recipe)


# 4. Setup tuning grid
prophet_uni_params <- prophet_uni_wflow %>%
  extract_parameter_set_dials(
    changepoint_num = changepoint_num(1, 100),
    changepoint_range = changepoint_range(0.6, 0.9),
    prior_scale_changepoints = prior_scale_changepoints(-3, 2),
    prior_scale_seasonality = prior_scale_seasonality(-3, 2),
    prior_scale_holidays = prior_scale_holidays(-3, 2)
  )
prophet_uni_grid <- grid_regular(prophet_uni_params, levels = 5)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(10)
registerDoParallel(cores.cluster)

prophet_uni_tuned <- tune_grid(
  prophet_uni_wflow,
  resamples = data_folds_uni,
  grid = prophet_uni_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)


prophet_uni_tuned %>% collect_metrics() %>%
  relocate(mean) %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
prophetsingle_autoplot <- autoplot(prophet_uni_tuned, metric = "rmse")
prophet_single_best <- show_best(prophet_uni_tuned, metric = "rmse")

# 7. Fitting Model

# changepoint_num = 50, changepoint_range = 0.75
# prior_scale_changepoints = 0.316, prior_scale_seasonality = 5.62, prior_scale_holidays = 0.001

prophet_uni_model <- prophet_reg(
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = TRUE, 
  seasonality_daily = FALSE,
  changepoint_num = 50, 
  changepoint_range = 0.75,
  prior_scale_changepoints = 0.316,
  prior_scale_seasonality = 5.62, 
  prior_scale_holidays = 0.001) %>%
  set_engine('prophet')

prophet_uni_recipe <- recipe(value ~ date + location,
                               data = train_uni_prophet) %>%
  step_dummy(all_nominal_predictors())


prophet_uni_wflow_tuned <- workflow() %>%
  add_model(prophet_uni_model) %>%
  add_recipe(prophet_uni_recipe)

prophet_uni_fit <- fit(prophet_uni_wflow_tuned, data = train_uni_prophet)



final_uni_train <- predict(prophet_uni_fit, new_data = train_uni_prophet) %>% 
  bind_cols(train_uni_prophet) %>% 
  rename(pred = .pred)
final_uni_test <- test_uni_prophet %>%
  bind_cols(predict(prophet_uni_fit, new_data = test_uni_prophet)) %>%
  rename(pred = .pred)


result_uni_train <- final_uni_train %>%
  group_by(location) %>%
  summarize(rmse_pred_train = ModelMetrics::rmse(value, pred)) %>%
  arrange(location)
result_uni_test <- final_uni_test %>%
  group_by(location) %>%
  summarize(rmse_pred_test = ModelMetrics::rmse(value, pred)) %>%
  arrange(location)

results_uni <- result_uni_train %>% 
  inner_join(result_uni_test, by = "location", suffix = c("rmse_train_pred", "rmse_test_pred"))

write.csv(results_uni, "Results/erica/prophet_uni/prophet_uni_rmse_results.csv", row.names = FALSE)


## Training + Testing Visualization

countries <- unique(final_uni_train$location)

for (loc in countries){
  train_title <- paste0("Training: Actual vs Predicted New Cases in ", loc, " in 2023")
  train_file <- paste0("Results/erica/prophet_uni/", loc, "_train_pred.jpeg")
  
  test_title <- paste0("Testing: Actual vs Predicted New Cases in ", loc, " in 2023")
  test_file <- paste0("Results/erica/prophet_uni/", loc, "_test_pred.jpeg")
  
  train_plot <- final_uni_train %>% 
    filter(location == loc) %>%
    ggplot(aes(x=date))+
    geom_line(aes(y = value, color = "Actual New Cases"))+
    geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")+
    scale_y_continuous(n.breaks = 15)+
    scale_x_date(date_breaks = "3 months", date_labels = "%b %y")+
    theme_minimal()+
    labs(x = "Date",
         y = "New Cases",
         title = train_title,
         subtitle = "prophet_reg(changepoint_num = 50, changepoint_range = 0.9,
       prior_scale_changepoints = 100, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.001)",
         caption = "Prophet Univariate",
         color = "")+
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  ggsave(train_plot, file = train_file, width=8, height =7, dpi = 300)
  
  test_plot <- final_uni_test %>% 
    filter(location == loc) %>% 
    ggplot(aes(x=date)) +
    geom_line(aes(y = value, color = "Actual New Cases")) +
    geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = test_title,
         subtitle = "prophet_reg(changepoint_num = 50, changepoint_range = 0.9,
       prior_scale_changepoints = 100, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.001",
         caption = "Prophet Univariate",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  ggsave(test_plot, file = test_file, width=8, height =7, dpi = 300)
  
}






