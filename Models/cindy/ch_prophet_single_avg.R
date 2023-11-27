library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(RcppRoll)

# Source
# https://rdrr.io/cran/modeltime/man/prophet_reg.html
# https://www.youtube.com/watch?v=kyPg3jV4pJ8 


# Setup parallel processing
cores <- detectCores()
cores.cluster <- makePSOCKcluster(5) 
registerDoParallel(cores.cluster)

# 1. Read in data ----
# NOTE: Using different data to accomodate countries that update weekly
train_lm <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm <- read_rds('data/avg_final_data/final_test_lm.rds')

# Making new data changes suggested from final presentation
# a. Remove observations before first appearance of COVID (2020-01-04)

# Get first covid observation for each country
first_covid <- train_lm |> 
  select(location, date, new_cases) |> 
  group_by(location) |> 
  filter(new_cases!= 0) |> 
  slice_min(order_by = row_number(), n = 1) |> 
  arrange(date)

# Calculate weekly rolling avg of new cases
complete_lm <- train_lm |> 
  bind_rows(test_lm) |> 
  filter(date >= as.Date("2020-01-04")) |> # first Covid obs
  group_by(date) |> 
  arrange(date, .by_group = TRUE) |> 
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA),
         value = ifelse(is.na(value), new_cases, value), 
         time_group = row_number(),
         seasonality_group = factor(row_number() %% 53)) |> 
  slice(seq(1, n(), by = 7)) |> 
  ungroup()

train_lm <- complete_lm |> filter(date < as.Date("2023-01-01"))
test_lm <- complete_lm |> filter(date >= as.Date("2023-01-01"))

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

prophet_recipe <- recipe(new_cases ~ date + location, data = train_lm) 
# NOTE TO SELF: needed to include location in recipe as well
#               since we are predicting on country level

# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_wflow <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

# 4. Setup tuning grid ----
# same parameters for both
prophet_params <- prophet_wflow |> 
  extract_parameter_set_dials()

prophet_grid <- grid_regular(prophet_params, levels = 3) 

# 5. Model Tuning ----
prophet_tuned <- tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

save(prophet_tuned, file = "Models/cindy/results/prophet_tuned_2.rda")

# 6. Review the best results ----
load("Models/cindy/results/prophet_tuned_2.rda")
show_best(prophet_tuned, metric = "rmse")

# changepoint_num changepoint_range prior_scale_seasonality prior_scale_holidays prior_scale_changepoints
# <int>             <dbl>                   <dbl>                <dbl>                    <dbl>
#             25               0.6                   0.001                0.001                    0.316
#             25               0.6                   0.001                0.316                    0.316
#             25               0.6                   0.001              100                        0.316
#             50               0.6                   0.001                0.001                    0.316
#             50               0.6                   0.001                0.316                    0.316

# changepoint_num = 25, changepoint_range = 0.6, prior_scale_changepoint = 0.903
#       prior_scale_holidays = 0.16, prior_scale_seasonality = 0.001

autoplot(prophet_tuned, metric = "rmse")

# 7. Fit best model ----
prophet_wflow_final <- prophet_wflow %>%
  finalize_workflow(select_best(prophet_tuned, metric = "rmse"))

prophet_fit <- fit(prophet_wflow_final, data = train_lm)

## Fitting with train data ----
prophet_train_results <- predict(prophet_fit, new_data = train_lm) |>
  bind_cols(train_lm) |>
  rename(pred = .pred)

prophet_train_results |> 
  group_by(location) |> 
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location) |>
  print(n = 23) |>
  view()


## Fitting with test data ----
prophet_pred <- predict(prophet_fit, new_data = test_lm) %>%
  bind_cols(test_lm %>% select(new_cases, location, date)) |>
  rename(pred = .pred)

prophet_result <- prophet_pred |>
  group_by(location) |>
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location) |> 
  print(n = 23) |>
  view()

# 8. Looking at final results ----
# Actual vs. Pred per country
unique_countries <- unique(train_lm$location)


## Training Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Training: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/prophet_single_avg/training_plots/prophet_single_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  prophet_country <- ggplot(prophet_train_results %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 25, changepoint_range = 0.6, 
         prior_scale_changepoint = 0.903, prior_scale_holidays = 0.16, prior_scale_seasonality = 0.001)",
         caption = "Prophet Univariate",
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
  file_name <- paste("Results/cindy/prophet_single_avg/testing_plots/prophet_single_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  prophet_country <- ggplot(prophet_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 25, changepoint_range = 0.6, 
         prior_scale_changepoint = 0.903, prior_scale_holidays = 0.16, prior_scale_seasonality = 0.001)",
         caption = "Prophet Univariate",
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
prophet_results <- prophet_pred %>%
  group_by(location) %>%
  summarise(value = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location) |> 
  view()


save(prophet_train_results, prophet_pred, prophet_results, file = "Models/cindy/results/prophet_single_avg.rda")




