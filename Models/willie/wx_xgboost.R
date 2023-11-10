library(tidyverse)
library(tidymodels)
library(doParallel)

# Source
# https://juliasilge.com/blog/xgboost-tune-volleyball/


# 1. Read in data
train_tree = readRDS('data/finalized_data/final_train_tree.rds')
test_tree = readRDS('data/finalized_data/final_test_tree.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds = rolling_origin(
  train_tree,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow
btree_model = boost_tree(
    trees = 1000, tree_depth = tune(),
    learn_rate = tune(), min_n = tune(), mtry = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
# View(btree_recipe %>% prep() %>% bake(new_data = NULL))

btree_wflow = workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid
btree_params = btree_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    min_n = min_n(c(5,15)),
    mtry = mtry(c(5,15)),
    tree_depth = tree_depth(c(2,20))
  )
btree_grid = grid_regular(btree_params, levels = 3)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)

btree_tuned = tune_grid(
  btree_wflow,
  resamples = data_folds,
  grid = btree_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

btree_tuned %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

# 6. Results
autoplot(btree_tuned, metric = "rmse")

# 7. Fit Best Model
# mtry = 15, min_n = 5, tree_depth = 20, learn_rate = 0.0178
btree_model = boost_tree(
  trees = 1000, tree_depth = 20,
  learn_rate = 0.0178, min_n = 5, mtry = 15) %>%
  set_engine('xgboost') %>%
  set_mode('regression')
btree_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
btree_wflow = workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

btree_fit = fit(btree_wflow, data = train_tree)
final_train = train_tree %>%
  bind_cols(predict(btree_fit, new_data = train_tree)) %>%
  rename(pred = .pred)

ggplot(final_train %>% filter(location == "United States")) +
  geom_line(aes(date, new_cases), color = 'red') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15)

library(ModelMetrics)
results = final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)


final_test = test_tree %>%
  bind_cols(predict(btree_fit, new_data = test_tree)) %>%
  rename(pred = .pred)
results_test = final_test %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)

