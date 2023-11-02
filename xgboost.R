library(tidyverse)
library(tidymodels)
library(doParallel)

# Source
# https://juliasilge.com/blog/xgboost-tune-volleyball/



# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(15)
registerDoParallel(cores.cluster)


# 1. Read in data
train_tree = readRDS('data/finalized_data/final_train_tree.rds')
test_tree = readRDS('data/finalized_data/final_test_tree.rds')

# 2. Create validation sets for every 3 months train + 1 month test with month increments
data_folds = rolling_origin(
  train_tree,
  initial = 30*6,
  assess = 30,
  skip = 30*3,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow
btree_model = boost_tree(
    trees = 1000, tree_depth = tune(),
    learn_rate = tune(), min_n = tune(), mtry = tune(),
    loss_reduction = tune(), sample_size = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_dummy(all_nominal_predictors())

btree_wflow = workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid





















