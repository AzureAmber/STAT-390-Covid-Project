library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)
# library(ModelMetrics)

# Source
# https://juliasilge.com


# Setup parallel processing
# detectCores() # 8
cores.cluster <- makePSOCKcluster(5)
registerDoParallel(cores.cluster)


# 1. Read in data ----
train_tree <- read_rds('data/finalized_data/final_train_tree.rds') |> 
  mutate(new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log))
test_tree <- read_rds('data/finalized_data/final_test_tree.rds') |> 
  mutate(new_cases_log = log(new_cases),
         new_cases_log = ifelse(new_cases_log == -Inf, 0, new_cases_log))

# 2. Create validation sets ----
# for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)

# 3. Define model, recipe, and workflow ----
btree_model <- boost_tree(
  # start with 500 trees first
  trees = 500, 
  tree_depth = tune(),
  learn_rate = tune(), 
  min_n = tune(), 
  mtry = tune(),
  # early stopping
  stop_iter = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

# using log(new_case) after arima log performed best
btree_recipe <- recipe(new_cases_log ~ ., data = train_tree) |> 
  step_rm(date) |> 
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0), 
    new_cases = ifelse(new_cases == -Inf, 0, new_cases)) |> 
  step_dummy(all_nominal_predictors())
# View(btree_recipe %>% prep() %>% bake(new_data = NULL))

btree_wflow <- workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid ----
btree_params <- btree_wflow %>%
  extract_parameter_set_dials() %>%
  # mtry is up to # of predictors (29)
  update(mtry = mtry(c(2, 29)),
         tree_depth = tree_depth(c(2,20)),
         stop_iter = stop_iter(c(10L,50L))
  )

btree_grid <- grid_regular(btree_params, levels = 3)

# 5. Model Tuning ----
tic.clearlog()
tic('xgboost_log')

btree_log_tuned = tune_grid(
  btree_wflow,
  resamples = data_folds,
  grid = btree_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)


toc(log = TRUE)
time_log <- tic.log(format = FALSE)
btree_log_tictoc <- tibble(model = time_log[[1]]$msg, 
                       runtime = time_log[[1]]$toc - time_log[[1]]$tic)
stopCluster(cores.cluster)


save(btree_log_tuned, btree_log_tictoc, file = "Models/cindy/results/btree_log_tuned_1.rda")

# 6. Review the best results ----
load("Models/cindy/results/btree_log_tuned_1.rda")
show_best(btree_log_tuned, metric = "rmse")
# mtry min_n tree_depth learn_rate stop_iter .metric .estimator  mean     n std_err .config               
# <int> <int>      <int>      <dbl>     <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                 
# 1    29     2         20     0.0178        10 rmse    standard   0.286   205  0.0110 Preprocessor1_Model048
# 2    29     2         11     0.0178        10 rmse    standard   0.287   205  0.0111 Preprocessor1_Model039
# 3    29     2         20     0.0178        50 rmse    standard   0.288   205  0.0115 Preprocessor1_Model210
# 4    29     2         11     0.0178        30 rmse    standard   0.288   205  0.0113 Preprocessor1_Model120
# 5    29     2         11     0.0178        50 rmse    standard   0.288   205  0.0114 Preprocessor1_Model201

autoplot(btree_log_tuned, metric = "rmse")


# 7. Fit Best Model ----
# mtry = 29, min_n = 2, tree_depth = 20, learn_rate = 0.0178, stop_iter = 10
# First using best model from above
# Creating new btree_model instead of finalize_workflow() so we can change trees
btree_model <- boost_tree(
  # Increasing from 500 --> 1000
  trees = 1000, 
  tree_depth = 20,
  learn_rate = 0.0178, 
  min_n = 2, 
  mtry = 29,
  # early stopping
  stop_iter = 10) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_wflow_final <- workflow() |> 
  add_model(btree_model) |> 
  add_recipe(btree_recipe)

btree_log_fit <- fit(btree_wflow_final, data = train_tree)

# Train data
btree_train_results <- predict(btree_log_fit, new_data = train_tree) %>%
  bind_cols(train_tree %>% select(new_cases, new_cases_log, location, date)) |>
  rename(pred = .pred) |> 
  # made sure to exp(pred) to undo log!!
  mutate(pred = exp(pred))

btree_train_results |>
  group_by(location) |>
  summarise(value = rmse(new_cases, pred)) |>
  arrange(location) |> 
  print(n = 23) |> 
  view()

# location         value
# <chr>            <dbl>
# 1 Argentina       13.2  
# 2 Australia      131.   
# 3 Canada           5.80 
# 4 Colombia         9.53 
# 5 Ecuador          1.20 
# 6 Ethiopia         0.774
# 7 France         140.   
# 8 Germany        122.   
# 9 India           63.5  
# 10 Italy           38.5  
# 11 Japan           66.3  
# 12 Mexico           9.48 
# 13 Morocco          2.10 
# 14 Pakistan         2.00 
# 15 Philippines      5.53 
# 16 Russia          32.6  
# 17 Saudi Arabia     1.11 
# 18 South Africa     5.63 
# 19 South Korea     51.4  
# 20 Sri Lanka        0.893
# 21 Turkey          39.4  
# 22 United Kingdom  32.2  
# 23 United States  257. 


btree_log_pred <- predict(btree_log_fit, new_data = test_tree) %>%
  bind_cols(test_tree %>% select(new_cases, new_cases_log, location, date)) |>
  rename(pred = .pred) |> 
  # made sure to exp(pred)!!!
  mutate(pred = exp(pred))

# 8. Looking at results ----
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
  file_name <- paste("Results/cindy/xgboost_log/training_plots/xgboost_log_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")

  btree_log_country <- ggplot(btree_train_results %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "boost_tree(mtry = 29, min_n = 2, tree_depth = 20, learn_rate = 0.0178, stop_iter = 10)",
         caption = "XGBoost Log",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))


  # Save the plot with specific dimensions
  ggsave(file_name, btree_log_country, width = 10, height = 6)
}

## Testing Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Testing: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/xgboost_log/testing_plots/xgboost_log_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  btree_log_country <- ggplot(btree_log_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         subtitle = "boost_tree(mtry = 29, min_n = 2, tree_depth = 20, learn_rate = 0.0178, stop_iter = 10)",
         caption = "XGBoost Log",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, btree_log_country, width = 10, height = 6)
}

# Table for test predictions
btree_log_results <- btree_log_pred |> 
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location) |> 
  print(n = 23)

# location           value
# <chr>              <dbl>
# 1 Argentina        784.   
# 2 Australia       1257.   
# 3 Canada           424.   
# 4 Colombia          56.2  
# 5 Ecuador           59.1  
# 6 Ethiopia           0.821
# 7 France          2157.   
# 8 Germany         4024.   
# 9 India            741.   
# 10 Italy           1090.   
# 11 Japan          11600.   
# 12 Mexico           377.   
# 13 Morocco           47.8  
# 14 Pakistan           1.60 
# 15 Philippines      144.   
# 16 Russia          1132.   
# 17 Saudi Arabia       3.33 
# 18 South Africa      86.0  
# 19 South Korea    15676.   
# 20 Sri Lanka          0.502
# 21 Turkey             1.00 
# 22 United Kingdom   434.   
# 23 United States  17244.   

# x <- btree_log_results |> 
#   pivot_wider(names_from = location, values_from = value)

save(btree_log_results, btree_log_pred, btree_train_results, file = "Models/cindy/results/btree_log.rda")

