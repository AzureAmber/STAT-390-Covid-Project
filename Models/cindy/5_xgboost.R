library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)

# Source
# https://juliasilge.com



# Setup parallel processing
# detectCores(logical = FALSE)
detectCores() # 8
cores.cluster <- makePSOCKcluster(4)
registerDoParallel(cores.cluster)


# 1. Read in data
train_tree <- read_rds('data/finalized_data/final_train_tree.rds')
test_tree <- read_rds('data/finalized_data/final_test_tree.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow
btree_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

btree_model <- boost_tree(
  # start with 500 trees first
  trees = 500, tree_depth = tune(),
  learn_rate = tune(), min_n = tune(), mtry = tune(),
  # early stopping
  stop_iter = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_wflow <- workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid
btree_params <- btree_wflow %>%
  extract_parameter_set_dials() %>%
  # mtry is up to # of predictors (31)
  update(mtry = mtry(c(2, 31)),
         tree_depth = tree_depth(c(2,20)),
         stop_iter = stop_iter(c(10L,50L))
  )

btree_grid <- grid_regular(btree_params, levels = 3)

# 5. Model Tuning
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

btree_tuned %>% collect_metrics() %>%
  group_by(.metric) |> 
  arrange(mean)

# mtry min_n tree_depth learn_rate stop_iter .metric .estimator  mean     n std_err .config               
# <int> <int>      <int>      <dbl>     <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                 
#   1     2    40          2      0.001        30 rsq     standard   0.610   156  0.0176 Preprocessor1_Model088
# 2     2    40          2      0.001        50 rsq     standard   0.610   157  0.0177 Preprocessor1_Model169
# 3     2    40         20      0.001        30 rsq     standard   0.611   166  0.0168 Preprocessor1_Model106
# 4     2    40         11      0.001        10 rsq     standard   0.612   161  0.0171 Preprocessor1_Model016
# 5     2    40          2      0.001        10 rsq     standard   0.614   162  0.0172 Preprocessor1_Model007
# 6     2    40         20      0.001        50 rsq     standard   0.617   163  0.0168 Preprocessor1_Model187
# 7     2    40         11      0.001        30 rsq     standard   0.618   170  0.0163 Preprocessor1_Model097
# 8     2    40         20      0.001        10 rsq     standard   0.619   161  0.0168 Preprocessor1_Model025
# 9     2    40         11      0.001        50 rsq     standard   0.620   156  0.0172 Preprocessor1_Model178
# 10    16    40         11      0.001        10 rsq     standard   0.651   155  0.0141 Preprocessor1_Model017

save(btree_tuned, btree_tictoc, file = "Models/cindy/results/btree_tuned_1.rda")

# NEXT STEPS: After finishing all 6 models, need to determine best of 6. THEN, fit the TESTING data to best model. 


