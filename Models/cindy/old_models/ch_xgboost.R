library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)
library(ModelMetrics)

# Source
# https://juliasilge.com


# Setup parallel processing
# detectCores() # 8
cores.cluster <- makePSOCKcluster(4)
registerDoParallel(cores.cluster)


# 1. Read in data ----
train_tree <- read_rds('data/finalized_data/final_train_tree.rds')
test_tree <- read_rds('data/finalized_data/final_test_tree.rds')

# 2. Create validation sets ----
# for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow ----
btree_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

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

btree_wflow <- workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid ----
btree_params <- btree_wflow %>%
  extract_parameter_set_dials() %>%
  # mtry is up to # of predictors (31)
  update(mtry = mtry(c(2, 31)),
         tree_depth = tree_depth(c(2,20)),
         stop_iter = stop_iter(c(10L,50L))
  )

btree_grid <- grid_regular(btree_params, levels = 3)

# 5. Model Tuning ----
tic.clearlog()
tic('xgboost')

btree_tuned = tune_grid(
  btree_wflow,
  resamples = data_folds,
  grid = btree_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse, rsq)
)


toc(log = TRUE)
time_log <- tic.log(format = FALSE)
btree_tictoc <- tibble(model = time_log[[1]]$msg, 
                       runtime = time_log[[1]]$toc - time_log[[1]]$tic)
stopCluster(cores.cluster)


save(btree_tuned, btree_tictoc, file = "Models/cindy/results/btree_tuned_1.rda")

# 6. Review the best results ----
load("Models/cindy/results/btree_tuned_1.rda")
show_best(btree_tuned, metric = "rmse")
# mtry min_n tree_depth learn_rate stop_iter .metric .estimator  mean     n std_err .config               
#   16     2          2     0.316         50 rmse    standard   7496.   163    853. Preprocessor1_Model218
#   31     2         11     0.0178        30 rmse    standard   7498.   156    975. Preprocessor1_Model120
#   31     2          2     0.316         50 rmse    standard   7528.   172   1001. Preprocessor1_Model219
#   16     2         11     0.0178        30 rmse    standard   7948.   154    956. Preprocessor1_Model119
#   31     2         20     0.0178        30 rmse    standard   7968.   169   1045. Preprocessor1_Model129

autoplot(btree_tuned, metric = "rmse")

# 7. Fit Best Model ----
# first using best model from above
# mtry = 16, min_n =2, tree_depth = 2, learn_rate = 0.316, stop_iter = 50
btree_wflow_final <- btree_wflow |> 
  finalize_workflow(select_best(btree_tuned, metric = "rmse"))

btree_fit <- fit(btree_wflow_final, data = train_tree)

# Train data 
btree_train_results <- predict(btree_fit, new_data = train_tree) %>% 
  bind_cols(train_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)
  # group_by(location) |> 
  # summarise(value = rmse(new_cases, pred)) |> 
  # arrange(location)

# location         value
# <chr>            <dbl>
# 1 Argentina       19387.
# 2 Australia       51840.
# 3 Canada           7803.
# 4 Colombia         9559.
# 5 Ecuador          1768.
# 6 Ethiopia          800.
# 7 France         176016.
# 8 Germany        158053.
# 9 India           83371.
# 10 Italy           40994.
# 11 Japan           57693.
# 12 Mexico          11840.
# 13 Morocco          2269.
# 14 Pakistan         2191.
# 15 Philippines      6720.
# 16 Russia          34375.
# 17 Saudi Arabia     1300.
# 18 South Africa     6309.
# 19 South Korea     70177.
# 20 Sri Lanka        1219.
# 21 Turkey          31682.
# 22 United Kingdom  38904.
# 23 United States  156766.

# Fitting with test data
btree_pred <- predict(btree_fit, new_data = test_tree) %>% 
  bind_cols(test_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)

# 8. Looking at results ----
# Actual vs. Pred per country
ggplot(btree_pred %>% filter(location == "Japan")) +
  geom_line(aes(date, new_cases), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "month") + 
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Actual vs. Predicted New Cases in United States in 2023") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# Table for test predictions
btree_results <- btree_pred %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)

# location            value
# <chr>               <dbl>
# 1 Argentina       5467.    
# 2 Australia      10035.    
# 3 Canada          2760.    
# 4 Colombia         519.    
# 5 Ecuador          435.    
# 6 Ethiopia          22.5   
# 7 France         15226.    
# 8 Germany        20497.    
# 9 India           2967.    
# 10 Italy           5033.    
# 11 Japan          44509.    
# 12 Mexico          2748.    
# 13 Morocco           85.4   
# 14 Pakistan          37.0   
# 15 Philippines      851.    
# 16 Russia          6723.    
# 17 Saudi Arabia     105.    
# 18 South Africa     437.    
# 19 South Korea    52976.    
# 20 Sri Lanka          3.22  
# 21 Turkey             0.0564
# 22 United Kingdom  2760.    
# 23 United States  65953. 

# x <- btree_results |> 
#   arrange(location) |> 
#   pivot_wider(names_from = location, values_from = value)

# 9. Improving performance ----
# Checking if poor performance is due to num of trees, learn_rate, etc.
autoplot(btree_tuned, metric = "rmse")

## Attempt 2 ----
# second best param combos
btree_model_2 <- boost_tree(
  trees = 1000, # increasing from 500 --> 1000 trees
  tree_depth = 11, 
  learn_rate = 0.0178, 
  min_n = 2, 
  mtry = 31,  
  stop_iter = 30) |>  
  set_engine('xgboost') |> 
  set_mode('regression')

btree_wflow_2 <- workflow() |> 
  add_model(btree_model_2) |> 
  # keeping same recipe from above
  add_recipe(btree_recipe)

btree_fit_2 <- fit(btree_wflow_2, data = train_tree)

# Train data 
predict(btree_fit_2, new_data = train_tree) %>% 
  bind_cols(train_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred) |> 
  group_by(location) |> 
  summarise(value = rmse(new_cases, pred)) |> 
  arrange(location) |> print(n = 23)

# NOTE: performed the same as attempt 1
# location         value
# <chr>            <dbl>
# 1 Argentina       19387.
# 2 Australia       51840.
# 3 Canada           7803.
# 4 Colombia         9559.
# 5 Ecuador          1768.
# 6 Ethiopia          800.
# 7 France         176016.
# 8 Germany        158053.
# 9 India           83371.
# 10 Italy           40994.
# 11 Japan           57693.
# 12 Mexico          11840.
# 13 Morocco          2269.
# 14 Pakistan         2191.
# 15 Philippines      6720.
# 16 Russia          34375.
# 17 Saudi Arabia     1300.
# 18 South Africa     6309.
# 19 South Korea     70177.
# 20 Sri Lanka        1219.
# 21 Turkey          31682.
# 22 United Kingdom  38904.
# 23 United States  156766.

# Test data
# NOTE: model may be severely overfitting to training set...
btree_pred_2 <- predict(btree_fit_2, new_data = test_tree) %>% 
  bind_cols(test_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)

btree_results_2 <- btree_pred_2 %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location) |> print(n = 23)

# btree_results_2 |> 
#   arrange(location) |> 
#   pivot_wider(names_from = location, values_from = value)

# NOTE: performed better on test than train, but exact same values as attempt 1
# location             value
# <chr>                <dbl>
# 1 Argentina       5467.     
# 2 Australia      10036.     
# 3 Canada          2760.     
# 4 Colombia         519.     
# 5 Ecuador          435.     
# 6 Ethiopia          22.6    
# 7 France         15226.     
# 8 Germany        20497.     
# 9 India           2967.     
# 10 Italy           5033.     
# 11 Japan          44509.     
# 12 Mexico          2748.     
# 13 Morocco           85.4    
# 14 Pakistan          36.7    
# 15 Philippines      851.     
# 16 Russia          6723.     
# 17 Saudi Arabia     105.     
# 18 South Africa     437.     
# 19 South Korea    52976.     
# 20 Sri Lanka          3.39   
# 21 Turkey             0.00124
# 22 United Kingdom  2760.     
# 23 United States  65953. 

## Attempt 3 ----
btree_model_3 <- boost_tree(
  trees = 1000, # increasing from 500 --> 1000 trees
  tree_depth = 11, 
  learn_rate = 0.0178, 
  min_n = 2, 
  mtry = 31) |>  
  set_engine('xgboost') |> 
  set_mode('regression')

btree_wflow_3 <- workflow() |> 
  add_model(btree_model_3) |> 
  # keeping same recipe from above
  add_recipe(btree_recipe)

btree_fit_3 <- fit(btree_wflow_3, data = train_tree)

# Train data 
predict(btree_fit_3, new_data = train_tree) %>% 
  bind_cols(train_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred) |> 
  group_by(location) |> 
  summarise(value = rmse(new_cases, pred)) |> 
  arrange(location) |> print(n = 23)

# SAME RESULTS AS ATTEMPT 1 and 2 ^^

# Test data
btree_pred_3 <- predict(btree_fit_3, new_data = test_tree) %>% 
  bind_cols(test_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)

btree_results_3 <- btree_pred_3 %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location) |> print(n = 23)

# SAME RESULTS AS ATTEMPT 1 and 2 ^^

# 10. Looking at final results ----
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
  file_name <- paste("Results/cindy/xgboost/training_plots/xgboost_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  xgboost_country <- ggplot(btree_train_results %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "boost_tree(trees = 500, mtry = 16, min_n =2, tree_depth = 2, learn_rate = 0.316, stop_iter = 50)",
         caption = "XGBoost",
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
  file_name <- paste("Results/cindy/xgboost/testing_plots/xgboost_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  xgboost_country <- ggplot(btree_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "boost_tree(trees = 500, mtry = 16, min_n =2, tree_depth = 2, learn_rate = 0.316, stop_iter = 50)",
         caption = "XGBoost",
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
btree_results <- btree_pred %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)

# location            value
# <chr>               <dbl>
# 1 Argentina       5467.    
# 2 Australia      10035.    
# 3 Canada          2760.    
# 4 Colombia         519.    
# 5 Ecuador          435.    
# 6 Ethiopia          22.5   
# 7 France         15226.    
# 8 Germany        20497.    
# 9 India           2967.    
# 10 Italy           5033.    
# 11 Japan          44509.    
# 12 Mexico          2748.    
# 13 Morocco           85.4   
# 14 Pakistan          37.0   
# 15 Philippines      851.    
# 16 Russia          6723.    
# 17 Saudi Arabia     105.    
# 18 South Africa     437.    
# 19 South Korea    52976.    
# 20 Sri Lanka          3.22  
# 21 Turkey             0.0564
# 22 United Kingdom  2760.    
# 23 United States  65953. 


save(btree_results, btree_train_results, btree_pred, file = "Models/cindy/results/btree.rda")



