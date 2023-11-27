library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(prophet)


# Source
# https://rdrr.io/cran/modeltime/man/prophet_reg.html


# Setup parallel processing
# cores <- detectCores()
cores.cluster <- makePSOCKcluster(5) 
registerDoParallel(cores.cluster)

# 1. Read in data ----
train_lm <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm <- read_rds('data/avg_final_data/final_test_lm.rds')


# 2. Create validation sets ----
# for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
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
             seasonality_daily = TRUE, # daily data
             season = "additive", # additive/multiplicative
             prior_scale_seasonality = tune(), # strength of seasonality model - larger fits more fluctuations
             prior_scale_holidays = tune(),# strength of holidays component
             prior_scale_changepoints = tune()) 

prophet_recipe <- recipe(new_cases ~ ., data = train_lm) |> 
  step_corr(all_numeric_predictors(), threshold = 0.7) |> 
  step_dummy(all_nominal_predictors()) 
# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_wflow <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

# 4. Setup tuning grid ----
# same parameters for both
prophet_params <- prophet_wflow |> 
  extract_parameter_set_dials() |> 
  update(
    changepoint_num = changepoint_num(range = c(5, 15)), 
    changepoint_range = changepoint_range(range = c(0.8, 0.95)), 
    prior_scale_seasonality = prior_scale_seasonality(range = c(10, 20)),
    prior_scale_holidays = prior_scale_holidays(range = c(5, 10))
  )

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
#              11             0.933                 8.61e12            95322704.                  0.00150
#              13             0.905                 2.29e10              252519.                  0.0186 
#               9             0.936                 2.37e17             3277605.                  0.0130 
#              15             0.846                 3.24e17             3011043.                  0.00625
#              13             0.803                 5.92e18              568747.                  0.0144 

autoplot(prophet_multi_tuned, metric = "rmse")

# 7. Fit best model ----
# changepoint_num = 11, changepoint_range = 0.933, prior_scale_seasonality = 8.61e12,
# prior_scale_holidays = 95322704., prior_scale_changepoints = 0.00150

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
prophet_multi_pred <- predict(prophet_multi_fit, new_data = test_lm) %>%
  bind_cols(test_lm %>% select(new_cases, location, date)) |>
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
  
  prophet_country <- ggplot(prophet_multi_train_results %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 17, changepoint_range = 0.704, 
         prior_scale_changepoints = 0.0107, prior_scale_seasonality = 37.2, prior_scale_holidays = 13.7)",
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
  
  prophet_country <- ggplot(prophet_multi_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 17, changepoint_range = 0.704, 
         prior_scale_changepoints = 0.0107, prior_scale_seasonality = 37.2, prior_scale_holidays = 13.7)",
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
# prophet_results <- prophet_pred %>%
#   group_by(location) %>%
#   summarise(value = ModelMetrics::rmse(new_cases, pred)) %>%
#   arrange(location) |> 
#   view()



# save(prophet_train_results, prophet_pred, prophet_results, file = "Models/cindy/results/prophet_single_avg.rda")




