library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll) # for rolling functions

# 1. Read in data ----
train_lm <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm <- read_rds('data/avg_final_data/final_test_lm.rds')

# Look for first COVID obs
train_lm |> 
  select(location, date, new_cases) |> 
  filter(new_cases != 0) |> 
  arrange(date)

# location      date       new_cases
# <chr>         <date>         <dbl>
# 1 Argentina     2020-01-01      3594 
# 2 Mexico        2020-01-01       991
# 3 Argentina     2020-01-02      3594
# 4 Mexico        2020-01-02       991
# 5 Germany       2020-01-04         1
# 6 Japan         2020-01-14         1
# 7 South Korea   2020-01-19         1
# 8 United States 2020-01-20         1
# 9 South Korea   2020-01-24         1
# 10 Australia     2020-01-25        1

# Argentina and Mexico don't start reporting until later

# Calculate weekly rolling avg of new cases
complete_lm <- train_lm |> 
  bind_rows(test_lm) |> 
  filter(date >= as.Date("2020-01-04")) |> # first Covid obs
  group_by(location) |> 
  arrange(date, .by_group = TRUE) |> 
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA),
         value = ifelse(is.na(value), new_cases, value)) |> 
  arrange(date, .by_group = TRUE) |> 
  slice(which(row_number() %% 7 == 0)) |> 
  mutate(time_group = row_number(),
         seasonality_group = row_number() %% 53) |> 
  ungroup() |> 
  mutate(seasonality_group = as.factor(seasonality_group))

train_lm <- complete_lm |> 
  filter(date < as.Date("2023-01-01")) |> 
  group_by(location) |> 
  arrange(date, .by_group = TRUE) |> 
  ungroup()

test_lm <- complete_lm |> 
  filter(date >= as.Date("2023-01-01")) |> 
  group_by(location) |> 
  arrange(date, .by_group = TRUE) |> 
  ungroup()

# 2. Model trend by country ----
# Seting up parallel processing
cores.cluster <- makePSOCKcluster(5) 
registerDoParallel(cores.cluster)

train_lm_fix <- tibble()
test_lm_fix <- tibble()

unique_countries <- unique(train_lm$location)

for (i in unique_countries) {
  data <- train_lm |> filter(location == i)
  complete_data <- complete_lm |> filter(location == i)
  
  max_time_group <- nrow(data) - 12
  data_for_lm <- data |> filter(time_group >= 13, time_group <= max_time_group)
  
  lm_model <- lm(value ~ 0 + time_group + seasonality_group, data = data_for_lm)
  x <- complete_data |>
    mutate(trend = predict(lm_model, newdata = complete_data),
           slope = coef(lm_model)["time_group"],
           seasonality_add = trend - slope * time_group,
           err = value - trend) |>
    mutate_if(is.numeric, round, 5)
  
  train_lm_fix <- bind_rows(train_lm_fix, x |> filter(date < as.Date("2023-01-01")))
  test_lm_fix <- bind_rows(test_lm_fix, x |> filter(date >= as.Date("2023-01-01")))
}


# 3. Build model ----
# For loop time...!
results <- tibble()
train_pred <- list()
test_pred <- list()


# Loop through each country
for (country in unique_countries) {
  # Subset data for the country
  train_lm_country <- train_lm_fix |> filter(location == country)
  test_lm_country <- test_lm_fix |> filter(location == country)
  
  data_folds <- rolling_origin(
    train_lm_country,
    initial = 53,
    assess = 4*2,
    skip = 4*4,
    cumulative = FALSE
  )
  
  autoarima_model <- arima_reg(seasonal_period = 53,
                               non_seasonal_ar = tune(), 
                               non_seasonal_differences = tune(), 
                               non_seasonal_ma = tune(),
                               seasonal_ar = tune(),
                               seasonal_differences = tune(), 
                               seasonal_ma = tune()) |> 
    set_engine("auto_arima")
  
  autoarima_recipe <- recipe(err ~ date, data = train_lm_country)
  
  autoarima_wflow <- workflow() |> 
    add_model(autoarima_model) |> 
    add_recipe(autoarima_recipe)
  
  autoarima_params <- autoarima_wflow |>
    extract_parameter_set_dials()
  
  autoarima_grid <- grid_regular(autoarima_params, levels = 5)
  
  autoarima_tuned <- tune_grid(
    autoarima_wflow,
    resamples = data_folds, 
    grid = autoarima_grid, 
    control = control_grid(save_pred = TRUE,
                           save_workflow = FALSE,
                           parallel_over = "everything"),
    metrics = metric_set(yardstick::rmse)
  )
  
  best_params <- select_best(autoarima_tuned, metric = "rmse")
  
  # Extract ARIMA parameters 
  best_pdq <- best_params$non_seasonal_ar |> toString() |>
    paste(best_params$non_seasonal_differences, best_params$non_seasonal_ma, sep = ", ")
  best_PDQ <- best_params$seasonal_ar |> toString() |>
    paste(best_params$seasonal_differences, best_params$seasonal_ma, sep = ", ")
  
  final_model <- finalize_workflow(autoarima_wflow, best_params) |>
    fit(data = train_lm_country)
  
  # Make predictions
  train_preds <- predict(final_model, new_data = train_lm_country)
  final_train <- train_lm_country |> 
    bind_cols(train_preds |> select(.pred)) |> 
    mutate(pred = trend + .pred)
  rmse_train <- ModelMetrics::rmse(final_train$value, final_train$pred)
  
  test_preds <- predict(final_model, new_data = test_lm_country)
  final_test <- test_lm_country |> 
    bind_cols(test_preds |> select(.pred)) |> 
    mutate(pred = trend + .pred)
  
  rmse_test <- ModelMetrics::rmse(final_test$value, final_test$pred)
  
  # Store results
  results <- bind_rows(results, tibble(
    country = country,
    rmse_train = rmse_train,
    rmse_test = rmse_test,
    pdq = best_pdq, 
    PDQ = best_PDQ
  ))
}

stopCluster(cores.cluster)


