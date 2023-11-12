library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)
library(ranger)
library(ModelMetrics)


# Setup parallel processing
# detectCores() # 8
cores.cluster <- makePSOCKcluster(6)
registerDoParallel(cores.cluster)


# 1. Read in data
train_tree_us <- read_rds('data/finalized_data/final_train_tree.rds') |> 
  filter(location == "United States")
test_tree_us <- read_rds('data/finalized_data/final_test_tree.rds') |> 
  filter(location == "United States")

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree_us,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)


# 3. Define model, recipe, and workflow
rf_us_model <- rand_forest(
  trees = 500,
  min_n = tune(), 
  mtry = tune()) |> 
  set_engine('ranger', importance = "impurity") %>%
  set_mode('regression')

rf_us_recipe <- recipe(new_cases ~ ., data = train_tree_us) |> 
  step_rm(date) |> 
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) |> 
  step_dummy(all_nominal_predictors(), -all_nominal(), -all_outcomes())  # exclude single-level factors

# View(rf_us_recipe %>% prep() %>% bake(new_data = NULL))

rf_us_wflow <- workflow() |> 
  add_model(rf_us_model) |> 
  add_recipe(rf_us_recipe)

# 4. Setup tuning grid
rf_us_params <- rf_us_wflow |> 
  extract_parameter_set_dials() |> 
  # mtry is up to # of predictors (29)
  update(mtry = mtry(c(2, 29)))

rf_us_grid <- grid_regular(rf_us_params, levels = 5)

# 5. Model Tuning
rf_us_tuned = tune_grid(
  rf_us_wflow,
  resamples = data_folds,
  grid = rf_us_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)


save(rf_us_tuned, file = "Models/cindy/results/rf_us_tuned.rda")

# 6. Review the best results (lower RMSE is better)
show_best(rf_us_tuned, metric = "rmse")
# mtry min_n .metric .estimator  mean     n std_err .config              
# <int> <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                
# 1    29    11 rmse    standard   5900.     6   2192. Preprocessor1_Model10
# 2    29     2 rmse    standard   5910.     6   2253. Preprocessor1_Model05
# 3    29    21 rmse    standard   6900.     6   2376. Preprocessor1_Model15
# 4    29    30 rmse    standard   8293.     6   2729. Preprocessor1_Model20
# 5    29    40 rmse    standard   9540.     6   2815. Preprocessor1_Model25

autoplot(rf_us_tuned, metric = "rmse")
# min_n = 11, mtry = 29

# #7. Fit best model
rf_us_model<- rand_forest(
# increasing number of trees from 500 --> 1000
  trees = 1000,
  min_n = 11,
  mtry = 29) |>
  set_engine('ranger', importance = "impurity") %>%
  set_mode('regression')

rf_us_wflow <- workflow() %>%
  add_model(rf_us_new) %>%
  add_recipe(rf_us_recipe)

rf_us_fit <- fit(rf_us_wflow, data = train_tree_us)

train_tree_us %>%
  bind_cols(predict(rf_us_fit, new_data = train_tree_us)) %>%
  rename(pred = .pred) |>
  group_by(location) |>
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location)
