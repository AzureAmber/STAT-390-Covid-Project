library(tidyverse)
library(tidymodels)
library(doParallel)



tidymodels_prefer()

# Source
# https://juliasilge.com



# Setup parallel processing

# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)


# 1. Read in data
train_tree <- readRDS('Data/finalized_data/final_train_tree.rds')
test_tree <- readRDS('Data/finalized_data/final_test_tree.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds

# 3. Define model, recipe, and workflow
btree_recipe <- recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())


btree_model <- boost_tree(
  trees = 2000, tree_depth = tune(),
  learn_rate = tune(), min_n = tune(), mtry = tune(),
  stop_iter = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_wflow <- workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid

# btree_grid <-
#   grid_latin_hypercube(
#     mtry(range = c(5L, 15L)),
#     tree_depth(range = c(2,20)),
#     stop_iter(range = c(10L, 50L)),
#     size = 3
#   )

btree_params <- btree_wflow %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1,5)),
         tree_depth = tree_depth(c(20,45)),
         stop_iter = stop_iter(c(10,50)),
         learn_rate = learn_rate(c(-1,0)),
         min_n = min_n(c(1,3))
  )

btree_grid <- grid_regular(btree_params, levels = 3)

# 5. Model Tuning


metric_set <- metric_set(yardstick::rmse, yardstick::rsq)

btree_tuned = tune_grid(
  btree_wflow,
  resamples = data_folds,
  grid = btree_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set
)

stopCluster(cores.cluster)

btree_tuned %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

save(btree_tuned, file = "Models/erica/results/btree_tuned_1.rda")

# 6. results
show_best(btree_tuned, metric = "rmse")


# btree_tuned %>% collect_metrics() %>%
#   group_by(.metric) %>%
#   arrange(mean)

btree_autoplot <- autoplot(btree_tuned, metric = "rmse")

jpeg("Models/erica/results/xgboost_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
print(btree_autoplot)
dev.off()



# # **FIX CODE BELOW**
# 
# # 7. Fit Best Model
# # Increase tree_depth, learn_rate, mtry
# # Decrease min_n
# btree_model <- boost_tree(
#   # increased number of trees
#   trees = 1000, 
#   tree_depth = 20,
#   learn_rate = 0.32, 
#   min_n = 2, 
#   mtry = 16) %>%
#   set_engine('xgboost') %>%
#   set_mode('regression')
# 
# btree_recipe <- recipe(new_cases ~ ., data = train_tree) %>%
#   step_rm(date) %>%
#   step_mutate(
#     G20 = ifelse(G20, 1, 0),
#     G24 = ifelse(G24, 1, 0)) %>%
#   step_dummy(all_nominal_predictors())
# 
# btree_wflow <- workflow() %>%
#   add_model(btree_model) %>%
#   add_recipe(btree_recipe)
# 
# btree_fit <- fit(btree_wflow, data = train_tree)
# 
# final_train <- train_tree %>%
#   bind_cols(predict(btree_fit, new_data = train_tree)) %>%
#   rename(pred = .pred)
# 
# ggplot(final_train %>% filter(location == "United States")) +
#   geom_line(aes(date, new_cases), color = 'red') +
#   geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
#   scale_y_continuous(n.breaks = 15)
# 
# 
# #predictions on country level
# results <- final_train %>%
#   group_by(location) %>%
#   summarise(value = rmse(new_cases, pred)) %>%
#   arrange(location)
# 
