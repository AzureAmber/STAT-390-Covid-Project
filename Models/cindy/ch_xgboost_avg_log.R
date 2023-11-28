#################################################################################
# NOTE: This is using the new average finalized data, plus adding lag variables.#
#################################################################################
set.seed(1024)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)

# Source
# https://juliasilge.com
# https://www.statology.org/r-lag/


# Setup parallel processing
# detectCores() # 8
cores.cluster <- makePSOCKcluster(5)
registerDoParallel(cores.cluster)


# 1. Read in data ----
# NOTE: Using different data to accomodate countries that update weekly
train_tree <- read_rds('data/avg_final_data/final_train_tree.rds')
test_tree <- read_rds('data/avg_final_data/final_test_tree.rds')

# Making new data changes suggested from final presentation
# a. Remove observations before first appearance of COVID (2020-01-04)

# Get first covid observation for each country
first_covid <- train_tree |> 
  select(location, date, new_cases) |> 
  group_by(location) |> 
  filter(new_cases!= 0) |> 
  slice_min(order_by = row_number(), n = 1) |> 
  arrange(date)

# location      date       new_cases
# <chr>         <date>         <dbl>
# 1 Argentina     2020-01-01      3594
# 2 Mexico        2020-01-01       991
# 3 Germany       2020-01-04         1
# 4 Japan         2020-01-14         1
# 5 South Korea   2020-01-19         1

train_tree |> 
  filter(location == "Argentina") |> 
  filter(new_cases != 0) |> 
  select(location, date, new_cases)

# location  date       new_cases
# <chr>     <date>         <dbl>
# 1 Argentina 2020-01-01      3594
# 2 Argentina 2020-01-02      3594
# 3 Argentina 2020-03-05        10

# NOTE: Argentina & Mexico don't actually start counting until later
# 2020-01-04 is "first" appearance of COVID

# b. Adding lagged predictors (1 day, 1 week, 1 month)
train_tree <- train_tree |> 
  filter(date >= as.Date("2020-01-04")) |> 
  mutate(one_day_lag = lag(new_cases, n = 1),
         one_week_lag = lag(new_cases, n = 7),
         one_month_lag = lag(new_cases, n = 30),
         new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log)) |> 
  group_by(date) |> 
  arrange(date, .by_group = TRUE) |> 
  ungroup()

test_tree <- test_tree |> 
  mutate(one_day_lag = lag(new_cases, n = 1),
         one_week_lag = lag(new_cases, n = 7),
         one_month_lag = lag(new_cases, n = 30),
         new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log)) |> 
  group_by(date) |> 
  arrange(date, .by_group = TRUE) |> 
  ungroup()

# 2. Create validation sets ----
# for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree,
  initial = 23*366,
  assess = 23*30*2,
  skip = 23*30*4,
  cumulative = FALSE
)

# 3. Define model, recipe, and workflow ----
btree_model <- boost_tree(
  trees = 1000, 
  tree_depth = tune(),
  learn_rate = tune(), 
  min_n = tune(), 
  mtry = tune(),
  # early stopping
  stop_iter = tune()) |> 
  set_engine('xgboost') |> 
  set_mode('regression')

btree_recipe <- recipe(new_cases_log ~ ., data = train_tree) |> 
  step_rm(date) |>
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) |>
  step_dummy(all_nominal_predictors())

# view(btree_recipe |> prep() |> bake(new_data = NULL))

btree_wflow <- workflow()  |> 
  add_model(btree_model) |>
  add_recipe(btree_recipe)


# 4. Setup tuning grid ----
btree_params <- btree_wflow |>
  extract_parameter_set_dials() |>
  # mtry is up to # of predictors 
  update(mtry = mtry(c(5, 28)),
         tree_depth = tree_depth(c(2,20)),
         stop_iter = stop_iter(c(10, 50)),
         min_n = min_n(c(5, 28))
  )

btree_grid <- grid_random(btree_params, size = 50)

# 5. Model Tuning ----
btree_log_tuned <- tune_grid(
  btree_wflow,
  resamples = data_folds,
  grid = btree_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse, mae)
)


stopCluster(cores.cluster)

save(btree_log_tuned, file = "Models/cindy/results/btree_log_tuned_2.rda")

# 6. Review the best results ----
load("Models/cindy/results/btree_log_tuned_2.rda")
show_best(btree_log_tuned, metric = "rmse")
# mtry min_n tree_depth learn_rate stop_iter .metric .estimator  mean     n std_err
# <int> <int>      <int>      <dbl>     <int> <chr>   <chr>      <dbl> <int>   <dbl>
#   28    18         16     0.0313        22 rmse    standard   0.127     6  0.0211
#   25    22          7     0.0558        16 rmse    standard   0.136     6  0.0226
#   25    10          3     0.0391        31 rmse    standard   0.139     6  0.0154
#   24    22         12     0.0225        22 rmse    standard   0.145     6  0.0235
#   27    14          2     0.0335        14 rmse    standard   0.146     6  0.0169

autoplot(btree_log_tuned, metric = "rmse")

# 7. Fit Best Model ----
# trees = 1000, mtry = 28, min_n = 18, tree_depth = 16, learn_rate = 0.0313, stop_iter = 22
btree_model <- boost_tree(
  trees = 1000,
  mtry = 28,
  min_n = 18,
  tree_depth = 16,
  learn_rate = 0.0313,
  stop_iter = 22) |>
  set_engine('xgboost') |>
  set_mode('regression')

btree_wflow_final <- workflow() |>
  add_model(btree_model) |>
  add_recipe(btree_recipe)

btree_fit <- fit(btree_wflow_final, data = train_tree)

## Fitting with train data ----
btree_train_results <- predict(btree_fit, new_data = train_tree) |>
  mutate(pred = exp(.pred)) |>
  bind_cols(train_tree |> select(new_cases, location, date))

btree_train_results |>
  group_by(location) |>
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location) |>
  print(n = 23) |>
  view()

## Fitting with test data ----
btree_pred <- predict(btree_fit, new_data = test_tree) |>
  mutate(pred = exp(.pred)) |> 
  bind_cols(test_tree |> select(new_cases, location, date))

# 8. Looking at results ----
# Actual vs. Pred per country
unique_countries <- unique(train_tree$location)

## Training Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Training: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/xgboost_avg_log/training_plots/xgboost_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  xgboost_country <- ggplot(btree_train_results |> filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "boost_tree(trees = 1000, mtry = 28, min_n = 18, tree_depth = 16, learn_rate = 0.0313, stop_iter = 22)",
         caption = "XGBoost Log",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))


  # Save the plot with specific dimensions
  ggsave(file_name, xgboost_country, width = 10, height = 6)
}

## Testing Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Testing: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/xgboost_avg_log/testing_plots/xgboost_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  xgboost_country <- ggplot(btree_pred |> filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "boost_tree(trees = 1000, mtry = 28, min_n = 18, tree_depth = 16, learn_rate = 0.0313, stop_iter = 22)",
         caption = "XGBoost Log",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))


  # Save the plot with specific dimensions
  ggsave(file_name, xgboost_country, width = 10, height = 6)
}

## Table for test predictions ----
btree_results <- btree_pred |>
  group_by(location) |>
  summarise(value = round(ModelMetrics::rmse(new_cases, pred), 4)) |>
  arrange(location) |>
  view()

# save(btree_results, btree_train_results, btree_pred, file = "Models/cindy/results/btree_avg_log.rda")



