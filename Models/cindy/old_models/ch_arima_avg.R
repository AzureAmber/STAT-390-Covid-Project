library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)
library(forecast)


# Setup parallel processing
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

automate_arima_process <- function(country_name, train_split, test_split) {
  # Extract train and test data for a specific country
  train_data <- train_split[[paste0("train_lm_fix_", country_name)]]
  test_data <- test_split[[paste0("test_lm_fix_", country_name)]]
  
  # Convert to time series
  train_ts <- train_data %>% select(err) %>% ts()
  
  # Initial ARIMA modeling
  auto_model <- auto.arima(train_ts)
  print(summary(auto_model)) # Prints the summary to check p, d, q values
  
  # ARIMA model setup
  arima_model <- arima_reg(
    seasonal_period = "auto",
    non_seasonal_ar = auto_model$arma[1], # p
    non_seasonal_differences = auto_model$arma[5], # d
    non_seasonal_ma = auto_model$arma[2], # q
    seasonal_ar = tune(),
    seasonal_differences = tune(),
    seasonal_ma = tune()
  ) |> 
    set_engine("arima")
  
  arima_recipe <- recipe(err ~ date, data = train_data) 
  
  # Workflow
  arima_wflow <- workflow() |>
    add_model(arima_model) |>
    add_recipe(arima_recipe)
  
  # Setup tuning grid
  arima_params <- extract_parameter_set_dials(arima_model)
  arima_grid <- grid_regular(arima_params, levels = 3)
  
  # Model Tuning
  arima_tuned <- tune_grid(
    arima_wflow,
    resamples = data_folds,
    grid = arima_grid,
    control = control_grid(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE)
  )
  
  # Best model parameters
  best_params <- select_best(arima_tuned, "rmse")
  
  # Fit best model
  final_arima_model <- finalize_model(arima_model, best_params)
  final_arima_wflow <- workflow() |> 
    add_model(final_arima_model) |> 
    add_recipe(arima_recipe)
  final_fit <- fit(final_arima_wflow, data = train_data)
  
  # Extract ARIMA parameters
  arima_params <- final_fit$fit$spec$spec$arma
  p <- arima_params[1]
  d <- arima_params[5]
  q <- arima_params[2]
  P <- arima_params[3]
  D <- arima_params[6]
  Q <- arima_params[4]
  
  # Predictions for training data
  train_data <- train_data |> 
    bind_cols(pred_err = final_fit$fit$fit$fit$data$.fitted) |> 
    mutate(pred = trend + pred_err) |> 
    mutate_if(is.numeric, round, 5)
  
  subtitle_text <- paste("arima_reg(seasonal_period = auto, (p,d,q) = (", p, ",", d, ",", q, "), (P,D,Q) = (", P, ",", D, ",", Q, "))", sep = "")
  
  # Check if 'new_cases' column exists in train_data and test_data
  if(!"new_cases" %in% names(train_data)) {
    stop("Column 'new_cases' not found in train_data")
  }
  if(!"new_cases" %in% names(test_data)) {
    stop("Column 'new_cases' not found in test_data")
  }
  
  # Plot for training data
  train_plot <- ggplot(train_data, aes(x = date)) +
    geom_line(aes(y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    labs(title = paste("Training: Actual vs Predicted New Cases in", country_name),
         subtitle = subtitle_text,
         x = "Date", 
         y = "New Cases",
         color = "",
         caption = "ARIMA") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

  
  ggsave(train_plot, file = tolower(paste0("Results/training_plots/arima_", make.names(country_name), ".jpeg")), width = 10, height = 6)
  
  
  # Predictions for testing data
  test_predictions <- augment(final_fit, new_data = test_data)
  test_data <- bind_cols(test_data, test_predictions)
  
  # Plot for testing data
  test_plot <- ggplot(test_data, aes(x = date)) +
    geom_line(aes(y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    labs(title = paste("Testing: Actual vs Predicted New Cases in", country_name),
         subtitle = subtitle_text,
         x = "Date", 
         y = "New Cases",
         color = "",
         caption = "ARIMA") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  # Save plot for testing data
  ggsave(test_plot, file = tolower(paste0("Results/testing_plots/arima_", make.names(country_name), ".jpeg")), width = 10, height = 6)
  
  saveRDS(final_fit, file = paste0("final_arima_model_", make.names(country_name), ".rds"))
  
}

# # Iterate over each country
# for (country in unique_countries) {
#   automate_arima_process(country, train_lm_fix_split, test_lm_fix_split)
# }

automate_arima_process("Germany", train_lm_fix_split, test_lm_fix_split)

stopCluster(cores.cluster)
