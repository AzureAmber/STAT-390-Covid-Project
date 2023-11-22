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
train_lm <- read_rds('data/finalized_data/final_train_lm.rds') |> 
  mutate(new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log))
test_lm <- read_rds('data/finalized_data/final_test_lm.rds') |> 
  mutate(new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log))


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

prophet_recipe <- recipe(new_cases_log ~ ., data = train_lm) |> 
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
prophet_multi_log_tuned <- tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)


stopCluster(cores.cluster)

save(prophet_multi_log_tuned, file = "Models/cindy/results/prophet_multi_log_tuned_1.rda")

# 6. Review the best results ----
load("Models/cindy/results/prophet_multi_log_tuned_1.rda")
show_best(prophet_multi_log_tuned, metric = "rmse")
# changepoint_num changepoint_range prior_scale_seasonality prior_scale_holidays prior_scale_changepoints .metric .estimator  mean     n
# <int>             <dbl>                   <dbl>                <dbl>                    <dbl> <chr>   <chr>      <dbl> <int>
# 1               0               0.6                 100                    0.001                      100 rmse    standard    1.20   205
# 2               0               0.9                 100                    0.001                      100 rmse    standard    1.20   205
# 3               0               0.6                 100                  100                          100 rmse    standard    1.20   205
# 4               0               0.9                 100                  100                          100 rmse    standard    1.20   205
# 5               0               0.6                   0.001                0.001                      100 rmse    standard    1.20   205

autoplot(prophet_multi_log_tuned, metric = "rmse")

# 7. Fit best model ----
# changepoint_num = 0, changepoint_range = 0.6, prior_scale_seasonality = 100, 
# prior_scale_holidays = 0.001, prior_scale_changepoints = 100
plog_wflow_final <- prophet_wflow |>
  finalize_workflow(select_best(prophet_multi_log_tuned, metric = "rmse"))

plog_fit <- fit(plog_wflow_final, data = train_lm)

# Train data
library(ModelMetrics)
plog_train_results <- predict(plog_fit, new_data = train_lm) |>
  bind_cols(train_lm |> select(new_cases, new_cases_log, location, date)) |>
  rename(pred = .pred) |> 
  # apply exp to undo log!!!
  mutate(pred = exp(pred)) 
  # group_by(location) |>
  # summarize(value = rmse(new_cases_log, pred)) |>
  # arrange(location) |>
  # print(n = 23)

# location         value
# <chr>            <dbl>
# 1 Argentina      1.76e 4
# 2 Australia      6.92e 9
# 3 Canada         5.90e 3
# 4 Colombia       8.03e 3
# 5 Ecuador        5.16e12
# 6 Ethiopia       7.52e 2
# 7 France         1.51e17
# 8 Germany        1.12e 9
# 9 India          3.49e 6
# 10 Italy          3.77e 4
# 11 Japan          4.08e 4
# 12 Mexico         1.60e 4
# 13 Morocco        1.94e 3
# 14 Pakistan       1.71e 3
# 15 Philippines    6.12e 3
# 16 Russia         4.34e 4
# 17 Saudi Arabia   1.07e 3
# 18 South Africa   4.35e 3
# 19 South Korea    1.90e 7
# 20 Sri Lanka      6.38e 2
# 21 Turkey         2.80e 4
# 22 United Kingdom 1.10e 5
# 23 United States  2.08e 6

plog_pred <- predict(plog_fit, new_data = test_lm) %>%
  bind_cols(test_lm %>% select(new_cases, location, date)) |>
  rename(pred = .pred) |> 
  # apply exp to undo log!!!
  mutate(pred = exp(pred))

# 8. Looking at test results ----
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
  file_name <- paste("Results/cindy/prophet_multi_log/training_plots/prophet_multi_log_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  plog_country <- ggplot(plog_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 0, changepoint_range = 0.6,
         prior_scale_changepoints = 100, prior_scale_seasonality = 100, prior_scale_holidays = 0.001)",
         caption = "Prophet Multivariate Log",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))


  # Save the plot with specific dimensions
  ggsave(file_name, plog_country, width = 10, height = 6)
}

## Testing Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Testing: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/prophet_multi_log/testing_plots/prophet_multi_log_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  plog_country <- ggplot(plog_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "prophet_reg(changepoint_num = 0, changepoint_range = 0.6,
         prior_scale_changepoints = 100, prior_scale_seasonality = 100, prior_scale_holidays = 0.001)",
         caption = "Prophet Multivariate Log",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, plog_country, width = 10, height = 6)
}

## Table for test predictions ----
plog_results <- plog_pred %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)

save(plog_train_results, plog_results, plog_pred, file = "Models/cindy/results/prophet_multi_log.rda")

