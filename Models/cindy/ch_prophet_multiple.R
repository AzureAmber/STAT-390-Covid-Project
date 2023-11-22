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
train_lm <- read_rds('data/finalized_data/final_train_lm.rds') 
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')


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
             seasonality_weekly = FALSE,
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
  extract_parameter_set_dials()

prophet_grid <- grid_regular(prophet_params, levels = 2) # had to decrease levels bc tuning was taking too long

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

save(prophet_multi_tuned, file = "Models/cindy/results/prophet_multi_tuned_1.rda")

# 6. Review the best results ----
load("Models/cindy/results/prophet_multi_tuned_1.rda")
show_best(prophet_multi_tuned, metric = "rmse")
# changepoint_num changepoint_range prior_scale_seasonality prior_scale_holidays prior_scale_changepo…¹ .metric .estimator   mean     n
# <int>             <dbl>                   <dbl>                <dbl>                  <dbl> <chr>   <chr>       <dbl> <int>
# 1               0               0.6                   0.001                0.001                    100 rmse    standard   11287.   205
# 2               0               0.9                   0.001                0.001                    100 rmse    standard   11287.   205
# 3               0               0.6                   0.001              100                        100 rmse    standard   11287.   205
# 4               0               0.9                   0.001              100                        100 rmse    standard   11287.   205
# 5               0               0.6                 100                    0.001                    100 rmse    standard   11298.   205

autoplot(prophet_multi_tuned, metric = "rmse")

# changepoint_num = 0.001, changepoint_range = 0.6, prior_scale_seasonality = 0.01

# 7. Fit best model ----
prophet_wflow_final <- prophet_wflow |> 
  finalize_workflow(select_best(prophet_multi_tuned, metric = "rmse"))

prophet_multi_fit <- fit(prophet_wflow_final, data = train_lm)

# Train data
library(ModelMetrics)
prophet_train_results <- predict(prophet_multi_fit, new_data = train_lm) |> 
  bind_cols(train_lm |> select(new_cases, location, date)) |> 
  rename(pred = .pred) 
  # group_by(location) |> 
  # summarize(value = rmse(new_cases, pred)) |> 
  # arrange(location)

# location        value
# <chr>           <dbl>
# 1 Argentina       8519.
# 2 Australia      62022.
# 3 Canada          5076.
# 4 Colombia        5818.
# 5 Ecuador        14174.
# 6 Ethiopia        3014.
# 7 France         18896.
# 8 Germany        41395.
# 9 India          46003.
# 10 Italy           5356.
# 11 Japan          24545.
# 12 Mexico         12565.
# 13 Morocco         3711.
# 14 Pakistan        3188.
# 15 Philippines     3999.
# 16 Russia         16825.
# 17 Saudi Arabia    3442.
# 18 South Africa    4735.
# 19 South Korea     9873.
# 20 Sri Lanka       3354.
# 21 Turkey          7465.
# 22 United Kingdom  8898.
# 23 United States  84084.

# Fitting with test data
prophet_multi_pred <- predict(prophet_multi_fit, new_data = test_lm) %>% 
  bind_cols(test_lm %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)

# 8. Looking at final results ----
# Actual vs. Pred per country
# Define the lists of countries
g20 <- c('Argentina', 'Australia', 'Canada', 'France', 'Germany',
         'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
         'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')

g24 <- c('Argentina', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
         'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')

# Combine and get unique countries from both lists
unique_countries <- unique(c(g20, g24))


## Training Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Training: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/prophet_multi/training_plots/prophet_multi_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  prophet_multi_country <- ggplot(prophet_train_results %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 0, changepoint_range = 0.6,
         prior_scale_changepoints = 100, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.001)",
         caption = "Prophet Multivariate",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, prophet_multi_country, width = 10, height = 6)
}

## Testing Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Testing: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/prophet_multi/testing_plots/prophet_multi_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  prophet_multi_country <- ggplot(prophet_multi_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 0, changepoint_range = 0.6,
         prior_scale_changepoints = 100, prior_scale_seasonality = 0.001, prior_scale_holidays = 0.001)",
         caption = "Prophet Multivariate",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, prophet_multi_country, width = 10, height = 6)
}

## Table for test predictions ----
prophet_multi_results <- prophet_multi_pred %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)


# location        value
# <chr>           <dbl>
# 1 Argentina       3807.
# 2 Australia      14287.
# 3 Canada          3644.
# 4 Colombia        2979.
# 5 Ecuador         2980.
# 6 Ethiopia        2535.
# 7 France          4070.
# 8 Germany         9587.
# 9 India          21083.
# 10 Italy           2923.
# 11 Japan          17882.
# 12 Mexico          5458.
# 13 Morocco         3427.
# 14 Pakistan        4240.
# 15 Philippines     3817.
# 16 Russia          5796.
# 17 Saudi Arabia    3426.
# 18 South Africa    2670.
# 19 South Korea     7005.
# 20 Sri Lanka       3065.
# 21 Turkey          2909.
# 22 United Kingdom  3744.
# 23 United States  51338.


save(prophet_train_results, prophet_multi_results, file = "Models/cindy/results/prophet_multi.rda")
