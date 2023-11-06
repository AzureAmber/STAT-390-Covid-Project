library(tidyverse)
library(tidymodels)
library(agua)
library(h2o)
h2o.init()
h2o_start()

# Source
# https://agua.tidymodels.org/articles/auto_ml.html
# https://jlaw.netlify.app/2022/05/03/ml-for-the-lazy-can-automl-beat-my-model/

# 1. Read in data
train_nn = readRDS('data/finalized_data/final_train_nn.rds')
test_nn = readRDS('data/finalized_data/final_test_nn.rds')

# 2. Define model, recipe, and workflow
# ***** NOTE: There are no tuning parameters: ?details_auto_ml_h2o
automl_model = auto_ml() %>%
  set_engine("h2o", max_runtime_secs = 120) %>%
  set_mode("regression")

automl_recipe = recipe(new_cases ~ ., data = train_nn) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
# View(automl_recipe %>% prep() %>% bake(new_data = NULL))

automl_wflow = workflow() %>%
  add_model(automl_model) %>%
  add_recipe(automl_recipe)

# 3. Fit Model
automl_fit = fit(automl_wflow, data = train_nn)

# Determine best model
automl_fit %>% collect_metrics()

automl_models = automl_fit %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  arrange(mean)
top_model = h2o.getModel(automl_models$id[1])

# Find model predictions
final_train = automl_recipe %>% prep() %>% bake(new_data = NULL)
automl_pred = h2o.predict(top_model, newdata = as.h2o(final_train)) %>%
  as_tibble() %>%
  rename(pred = predict) %>%
  bind_cols(train_nn)

library(ModelMetrics)
results = automl_pred %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)

