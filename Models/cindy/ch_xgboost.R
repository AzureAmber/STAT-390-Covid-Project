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
btree_wflow_final <- btree_wflow |> 
  finalize_workflow(select_best(btree_tuned, metric = "rmse"))

btree_fit <- fit(btree_wflow_final, data = train_tree)

# Train data 
predict(btree_fit, new_data = train_tree) %>% 
  bind_cols(train_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred) |> 
  group_by(location) |> 
  summarise(value = rmse(new_cases, pred)) |> 
  arrange(value)

# location        value
# 1 Morocco         1016.
# 2 Saudi Arabia    1062.
# 3 Ethiopia        1225.
# 4 Sri Lanka       1478.
# 5 Pakistan        1511.
# 6 South Africa    1933.
# 7 Colombia        1982.
# 8 Ecuador         2379.
# 9 Argentina       2624.
# 10 Philippines     2784.
# 11 Canada          2786.
# 12 Mexico          2939.
# 13 Turkey          3165.
# 14 Russia          3786.
# 15 Italy           3862.
# 16 United Kingdom  4144.
# 17 Australia       5457.
# 18 Japan           6058.
# 19 South Korea     6470.
# 20 Germany         8431.
# 21 France          8465.
# 22 India           8869.
# 23 United States  12087.

btree_pred <- predict(btree_fit, new_data = test_tree) %>% 
  bind_cols(test_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)

# 8. Looking at results ----
# Actual vs. Pred per country
ggplot(btree_pred %>% filter(location == "United States")) +
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
  arrange(value)

# location         value
# 1 Sri Lanka         453.
# 2 Ethiopia          503.
# 3 Pakistan          908.
# 4 Turkey           1099.
# 5 Colombia         1181.
# 6 South Africa     1736.
# 7 Saudi Arabia     1803.
# 8 Ecuador          1852.
# 9 Canada           2374.
# 10 Russia           5290.
# 11 Germany          6425.
# 12 France           9383.
# 13 Philippines     10976.
# 14 Mexico          15059.
# 15 South Korea     16241.
# 16 Australia       18255.
# 17 Morocco         18865.
# 18 Japan           20802.
# 19 India           25865.
# 20 Argentina       29811.
# 21 United Kingdom  64916.
# 22 Italy           86135.
# 23 United States  348243.

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
  arrange(value) |> print(n = 23)

# location       value
# <chr>          <dbl>
# 1 Saudi Arabia    27.2
# 2 Sri Lanka       30.6
# 3 Ethiopia        31.9
# 4 Morocco         44.9
# 5 Ecuador         52.7
# 6 Pakistan        64.6
# 7 Germany         72.1
# 8 Canada          75.4
# 9 South Africa    75.6
# 10 France          83.0
# 11 Australia       84.1
# 12 Argentina       89.9
# 13 Colombia        93.5
# 14 Mexico         100. 
# 15 Philippines    104. 
# 16 South Korea    126. 
# 17 Russia         142. 
# 18 Turkey         145. 
# 19 United Kingdom 147. 
# 20 Japan          164. 
# 21 Italy          168. 
# 22 India          194. 
# 23 United States  213. 

# Test data
# NOTE: model may be severely overfitting to training set...
btree_pred_2 <- predict(btree_fit_2, new_data = test_tree) %>% 
  bind_cols(test_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)

btree_results_2 <- btree_pred_2 %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(value) |> print(n = 23)

btree_results_2 |> 
  arrange(location) |> 
  pivot_wider(names_from = location, values_from = value)
# NOTE: not performing well at all...need to change parameter values
# location          value
# <chr>             <dbl>
# 1 Saudi Arabia       16.5
# 2 Ethiopia           25.1
# 3 Sri Lanka          59.2
# 4 Pakistan           79.8
# 5 South Africa      150. 
# 6 Colombia          402. 
# 7 Canada            858. 
# 8 Ecuador          3015. 
# 9 Mexico           3046. 
# 10 Turkey           3768. 
# 11 Russia           7355. 
# 12 Philippines     11439. 
# 13 Germany         12924. 
# 14 Morocco         14677. 
# 15 France          16702. 
# 16 Japan           17929. 
# 17 South Korea     23957. 
# 18 Australia       35333. 
# 19 United Kingdom  38927. 
# 20 India           87147. 
# 21 Argentina       96022. 
# 22 Italy          110113. 
# 23 United States  573818. 

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
  arrange(value) |> print(n = 23)

# Test data
btree_pred_3 <- predict(btree_fit_3, new_data = test_tree) %>% 
  bind_cols(test_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred)

btree_results_3 <- btree_pred_3 %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(value) |> print(n = 23)



