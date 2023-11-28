library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)

set.seed(1024)
# Source
# https://rdrr.io/cran/modeltime/man/prophet_reg.html


# Setup parallel processing
# cores <- detectCores()
cores.cluster <- makePSOCKcluster(5) 
registerDoParallel(cores.cluster)

# 1. Read in data ----
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

train_lm <- complete_lm |> filter(date < as.Date("2023-01-01")) |>
  group_by(date) |>
  arrange(date, .by_group = TRUE) |>
  ungroup()

test_lm <- complete_lm |> filter(date >= as.Date("2023-01-01")) |>
  group_by(date) |>
  arrange(date, .by_group = TRUE) |>
  ungroup()
  
# 2. Create validation sets ----
# for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
    train_lm,
    initial = 23*53,
    assess = 23*4*2,
    skip = 23*4*4,
    cumulative = FALSE
  )

# 3. Define model, recipe, and workflow ----
prophet_model <- prophet_reg() |> 
  set_engine("prophet", 
             growth = "linear", # linear/logistic 
             changepoint_num = tune(), # num of potential changepoints for trend
             changepoint_range = tune(), # adjusts flexibility of trend
             seasonality_yearly = FALSE, 
             seasonality_weekly = TRUE,
             seasonality_daily = FALSE,
             season = "additive", # additive/multiplicative
             prior_scale_seasonality = tune(), # strength of seasonality model - larger fits more fluctuations
             prior_scale_holidays = tune(),# strength of holidays component
             prior_scale_changepoints = tune()) 

prophet_recipe <- recipe(new_cases ~ ., data = train_lm) |> 
  # had to remove day_of_week or else couldn't bake
  step_rm(day_of_week) |> 
  step_corr(all_numeric_predictors(), threshold = 0.7) |> 
  step_dummy(all_nominal_predictors()) 
# View(prophet_recipe |> prep() |> bake(new_data = NULL))

prophet_wflow <- workflow() |>
  add_model(prophet_model) |>
  add_recipe(prophet_recipe)

# 4. Setup tuning grid ----
# same parameters for both
prophet_params <- prophet_wflow |> 
  extract_parameter_set_dials()

prophet_grid <- grid_random(prophet_params, size = 50) 

# 5. Model Tuning ----
prophet_multi_tuned <- tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)


stopCluster(cores.cluster)

save(prophet_multi_tuned, file = "Models/cindy/results/prophet_multi_tuned_2.rda")

# 6. Review the best results ----
load("Models/cindy/results/prophet_multi_tuned_2.rda")
show_best(prophet_multi_tuned, metric = "rmse")
# changepoint_num changepoint_range prior_scale_seasonality prior_scale_holidays prior_scale_changepoints
# <int>             <dbl>                   <dbl>                <dbl>                    <dbl>
#              36             0.735                 3.92                 4.54                     0.00142
#              31             0.634                 3.15                71.1                      0.00129
#              25             0.703                 4.58                 1.35                     0.0500 
#              13             0.619                 0.106                0.00101                  0.00149
#              27             0.808                 0.00150              0.00349                 59.2  

autoplot(prophet_multi_tuned, metric = "rmse")

# 7. Fit best model ----
# changepoint_num = 36, changepoint_range = 0.735, prior_scale_seasonality = 3.92,
# prior_scale_holidays = 4.54, prior_scale_changepoints = 0.00142

prophet_wflow_final <- prophet_wflow |>
  finalize_workflow(select_best(prophet_multi_tuned, metric = "rmse"))

prophet_multi_fit <- fit(prophet_wflow_final, data = train_lm)

## Fitting with train data ----
prophet_multi_train_results <- predict(prophet_multi_fit, new_data = train_lm) |>
  bind_cols(train_lm) |>
  rename(pred = .pred)

prophet_multi_train_results |>
  group_by(location) |>
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location) |>
  print(n = 23) |>
  view()

## Fitting with test data ----
prophet_multi_pred <- predict(prophet_multi_fit, new_data = test_lm) |>
  bind_cols(test_lm |> select(new_cases, location, date)) |>
  rename(pred = .pred)

# 8. Looking at final results ----
# Actual vs. Pred per country
unique_countries <- unique(train_lm$location)


## Training Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Training: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/prophet_multi_avg/training_plots/prophet_multi_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  prophet_country <- ggplot(prophet_multi_train_results |> filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 36, changepoint_range = 0.735, 
         prior_scale_seasonality = 3.92, prior_scale_holidays = 4.54, prior_scale_changepoints = 0.00142)",
         caption = "Prophet Multivariate",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))


  # Save the plot with specific dimensions
  ggsave(file_name, prophet_country, width = 10, height = 6)
}

## Testing Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Testing: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/prophet_multi_avg/testing_plots/prophet_multi_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  prophet_country <- ggplot(prophet_multi_pred |> filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 36, changepoint_range = 0.735, 
         prior_scale_seasonality = 3.92, prior_scale_holidays = 4.54, prior_scale_changepoints = 0.00142)",
         caption = "Prophet Multivariate",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))


  # Save the plot with specific dimensions
  ggsave(file_name, prophet_country, width = 10, height = 6)
}

## Table for test predictions ----
prophet_multi_results <- prophet_multi_pred |>
  group_by(location) |>
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location) |>
  view()



# save(prophet_multi_train_results, prophet_multi_pred, prophet_multi_results, file = "Models/cindy/results/prophet_multi_avg.rda")




