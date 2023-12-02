library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)
library(forecast)

tidymodels_prefer()

# Source
# https://www.rdocumentation.org/packages/modeltime/versions/1.2.8/topics/arima_reg
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/


# Setup parallel processing
# detectCores() # 8
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm <- read_rds('data/avg_final_data/final_test_lm.rds')


complete_lm <- train_lm |> 
  bind_rows(test_lm) |> 
  filter(date >= as.Date("2020-01-04")) |>
  group_by(location) |>
  arrange(date, .by_group = TRUE) |>
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA)) |>
  mutate(value = ifelse(is.na(value), new_cases, value)) |>
  arrange(date, .by_group = TRUE) |>
  slice(which(row_number() %% 7 == 0)) |>
  mutate(
    time_group = row_number(),
    seasonality_group = row_number() %% 53) |>
  ungroup() |>
  mutate(seasonality_group = as.factor(seasonality_group))

train_lm <- complete_lm |> 
  filter(date < as.Date("2023-01-01")) |>
  group_by(date) |>
  arrange(date, .by_group = TRUE) |>
  ungroup()

test_lm <- complete_lm |> 
  filter(date >= as.Date("2023-01-01")) |>
  group_by(date) |>
  arrange(date, .by_group = TRUE) |>
  ungroup()

# STATIONARITY CHECK IN arima_notes.R

# 2. Find each country model trend
train_lm_fix <- tibble()
test_lm_fix <- tibble()

unique_countries <- unique(train_lm$location)

for (country in unique_countries) {
  data <- train_lm |> filter(location == country)
  complete_data <- complete_lm |> filter(location == country)
  
  country_data = 
  # Find linear model
  lm_model <- lm(value ~ 0 + time_group + seasonality_group,
                data |> filter(between(time_group, 13, nrow(data) - 12)))
  
  x <- complete_data |>
    mutate(
      trend = predict(lm_model, newdata = complete_data),
      slope = as.numeric(coef(lm_model)["time_group"]),
      seasonality_add = trend - slope * time_group,
      err = value - trend)|>
    mutate_if(is.numeric, round, 5)
  train_lm_fix <<- rbind(train_lm_fix, x |> filter(date < as.Date("2023-01-01")))
  test_lm_fix <<- rbind(test_lm_fix, x |> filter(date >= as.Date("2023-01-01")))
}

# STATIONARITY CHECK FOR LINEAR IN arima_notes.R

# Splitting data and storing in a list
split_data <- function(data, prefix) {
  split_list <- lapply(setNames(nm = unique_countries), function(loc) {
    data %>% filter(location == loc)
  })
  names(split_list) <- paste0(prefix, make.names(unique_countries))
  return(split_list)
}

# Apply the function to both datasets
train_lm_fix_split <- split_data(train_lm_fix, "train_lm_fix_")
test_lm_fix_split <- split_data(test_lm_fix, "test_lm_fix_")


# 3. Define model, recipe, and workflow
data_folds <- rolling_origin(
  train_lm_fix,
  initial = 53,
  assess = 4*2,
  skip = 4*4,
  cumulative = FALSE
)

# 3. Data for each country ----
train_data <- train_lm_fix_split$train_lm_fix_Saudi.Arabia
test_data <- test_lm_fix_split$test_lm_fix_Saudi.Arabia


# Get pdq
train_data |> 
  select(err) |> 
  ts() |> 
  auto.arima() |> 
  summary() 


arima_model <- arima_reg(
  seasonal_period = "auto", # default
  non_seasonal_ar = 1, # p (0-5)
  non_seasonal_differences = 1, # d (0-2)
  non_seasonal_ma = 1, # q (0-5)
  seasonal_ar = tune(), 
  seasonal_differences = tune(), 
  seasonal_ma = tune()
) |> 
  set_engine("arima")

arima_recipe <- recipe(err ~ date, data = train_data) 

arima_wflow <- workflow()|>
  add_model(arima_model)|>
  add_recipe(arima_recipe)

# 4. Setup tuning grid ----
arima_params <- arima_wflow |> 
  extract_parameter_set_dials()

arima_grid <- grid_regular(arima_params, levels = 3)

# 5. Model Tuning ----
arima_tuned <- tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)


# 6. Results ----
show_best(arima_tuned, metric = "rmse")


# 7. Fit best model ----
# arima_model <- arima_reg(
#   seasonal_period = "auto", # default
#   non_seasonal_ar = 5, # p (0-5)
#   non_seasonal_differences = 0, # d (0-2)
#   non_seasonal_ma = 0, # q (0-5)
#   seasonal_ar = 0,
#   seasonal_differences = 0,
#   seasonal_ma = 0
# ) |>
#   set_engine("arima")
# 
# arima_wflow <- workflow() |>
#   add_model(arima_model) |>
#   add_recipe(arima_recipe)
# 
# arima_fit <- fit(arima_wflow, data = train_data)
# 
# ## Fitting with train data ----
# final_train <- train_data |>
#   bind_cols(pred_err = arima_fit$fit$fit$fit$data$.fitted) |>
#   mutate(pred = trend + pred_err) |>
#   mutate_if(is.numeric, round, 5)
# 
# arima_train_plot <- ggplot(final_train |> filter(location == "Saudi Arabia"), aes(x = date)) +
#   geom_line(aes(y = new_cases, color = "Actual New Cases")) +
#   geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")  +
#   scale_y_continuous(n.breaks = 15) +
#   scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
#   theme_minimal() +
#   labs(x = "Date",
#        y = "New Cases",
#        title = "Training: Actual vs. Predicted New Cases in Saudi Arabia",
#        subtitle = "arima_reg(seasonal_period = auto, (p,d,q) = (5,0,0), (P,D,Q) = (0,0,0))",
#        caption = "ARIMA",
#        color = "") +
#   theme(plot.title = element_text(face = "bold", hjust = 0.5),
#         plot.subtitle = element_text(face = "italic", hjust = 0.5),
#         legend.position = "bottom",
#         panel.grid.minor = element_blank()) +
#   scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
# 
# ggsave(arima_train_plot, file = "Results/cindy/arima_avg/training_plots/arima_saudi_arabia.jpeg",  width = 10, height = 6)
# 
# 
# # rmse of error prediction
# ModelMetrics::rmse(final_train$err, final_train$pred_err)
# # rmse of just linear trend
# ModelMetrics::rmse(final_train$value, final_train$trend)
# # rmse of linear trend + arima
# ModelMetrics::rmse(final_train$value, final_train$pred)
# 
# ## Fitting with test data ----
# final_test <- test_data |>
#   bind_cols(predict(arima_fit, new_data = test_data)) |>
#   mutate(pred_err = .pred) |>
#   mutate(pred = trend + pred_err) |>
#   mutate_if(is.numeric, round, 5)
# 
# arima_test_plot <- ggplot(final_test |> filter(location == "Saudi Arabia"), aes(x = date)) +
#   geom_line(aes(y = new_cases, color = "Actual New Cases")) +
#   geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")  +
#   scale_y_continuous(n.breaks = 15) +
#   scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
#   theme_minimal() +
#   labs(x = "Date",
#        y = "New Cases",
#        title = "Testing: Actual vs. Predicted New Cases in Saudi Arabia",
#        subtitle = "arima_reg(seasonal_period = auto, (p,d,q) = (5,0,0), (P,D,Q) = (0,0,0))",
#        caption = "ARIMA",
#        color = "") +
#   theme(plot.title = element_text(face = "bold", hjust = 0.5),
#         plot.subtitle = element_text(face = "italic", hjust = 0.5),
#         legend.position = "bottom",
#         panel.grid.minor = element_blank()) +
#   scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
# 
# ggsave(arima_test_plot, file = "Results/cindy/arima_avg/testing_plots/arima_saudi_arabia.jpeg",  width = 10, height = 6)
# 
# 
# # rmse of error prediction
# ModelMetrics::rmse(final_test$err, final_test$pred_err)
# # rmse of just linear trend
# ModelMetrics::rmse(final_test$value, final_test$trend)
# # rmse of linear trend + arima
# ModelMetrics::rmse(final_test$value, final_test$pred)

