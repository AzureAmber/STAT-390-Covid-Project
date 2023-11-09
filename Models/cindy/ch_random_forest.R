library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)
library(ranger)


# Setup parallel processing
# detectCores() # 8
cores.cluster <- makePSOCKcluster(5)
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


# 3. Define model, recipe, and workflow
rf_model <- rand_forest(
  # start with 500 trees instead of tuning (match w bt)
  trees = 500,
  min_n = tune(), 
  mtry = tune()) |> 
  set_engine('ranger', importance = "impurity") %>%
  set_mode('regression')

rf_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  # will later add normalizing to improve perf
  step_dummy(all_nominal_predictors())

# View(rf_recipe %>% prep() %>% bake(new_data = NULL))

rf_wflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rf_recipe)

# 4. Setup tuning grid
rf_params <- rf_wflow %>%
  extract_parameter_set_dials() %>%
  # mtry is up to # of predictors (30)
  update(mtry = mtry(c(5, 30)))

rf_grid <- grid_regular(rf_params, levels = 3)

# 5. Model Tuning
tic.clearlog()
tic('rf')

rf_tuned = tune_grid(
  rf_wflow,
  resamples = data_folds,
  grid = rf_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)


toc(log = TRUE)
time_log <- tic.log(format = FALSE)
rf_tictoc <- tibble(model = time_log[[1]]$msg, 
                       runtime = time_log[[1]]$toc - time_log[[1]]$tic)
stopCluster(cores.cluster)


save(rf_tuned, rf_tictoc, file = "Models/cindy/results/rf_tuned_1.rda")

# 6. Review the best results (lower RMSE is better)
show_best(rf_tuned, metric = "rmse")
# mtry min_n .metric .estimator   mean     n std_err .config             
#   30     2 rmse    standard    9638.   205   1147. Preprocessor1_Model3
#   17     2 rmse    standard   10690.   205   1215. Preprocessor1_Model2
#   30    21 rmse    standard   13689.   205   1590. Preprocessor1_Model6
#   17    21 rmse    standard   14852.   205   1616. Preprocessor1_Model5
#   30    40 rmse    standard   15231.   205   1683. Preprocessor1_Model9

# NOTE: big mtry, small min_n, increase trees 2nd round!
autoplot(rf_tuned, metric = "rmse")

#7. Fit best model

rf_model_new <- rand_forest(
# increasing number of trees from 500 --> 1000
  trees = 100,
  min_n = 2, 
  mtry = 30) |> 
  set_engine('ranger', importance = "impurity") %>%
  set_mode('regression')


rf_recipe_new <- recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors()) |> 
  # adding new steps to recipe 
  step_normalize(all_numeric_predictors()) |> 
  step_zv(all_predictors())

rf_wflow_new <- workflow() %>%
  add_model(rf_model_new) %>%
  add_recipe(rf_recipe_new)


rf_fit <- fit(rf_wflow_new, data = train_tree)

rf_final_train <- train_tree %>%
  bind_cols(predict(rf_fit, new_data = train_tree)) %>%
  rename(pred = .pred)

ggplot(rf_final_train %>% filter(location == "United States")) +
  geom_line(aes(date, new_cases), color = 'black') +
  geom_line(aes(date, pred), color = 'blue', linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "New Covid Cases for US")

ggsave('Models/cindy/plots/us_rf_plot.jpg')

rf_final_train |> 
  group_by(location) |> 
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |> 
  arrange(value) 

# location         value
# 1 Saudi Arabia      15.0
# 2 Ethiopia          20.6
# 3 Morocco           42.6
# 4 Pakistan          50.9
# 5 South Africa     144. 
# 6 Philippines      155. 
# 7 Colombia         203. 
# 8 Mexico           314. 
# 9 Russia           443. 
# 10 Sri Lanka        461. 
# 11 Italy            598. 
# 12 United Kingdom   712. 
# 13 Argentina        931. 
# 14 Turkey          1414. 
# 15 Japan           2000. 
# 16 South Korea     2471. 
# 17 Canada          2603. 
# 18 India           2936. 
# 19 Australia       3511. 
# 20 United States   6518. 
# 21 Germany         6572. 
# 22 Ecuador         6945. 
# 23 France         11609. 




