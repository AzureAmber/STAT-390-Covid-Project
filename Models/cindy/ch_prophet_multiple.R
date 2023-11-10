library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(prophet)


# Source
# https://rdrr.io/cran/modeltime/man/prophet_reg.html


# Setup parallel processing
# cores <- detectCores()
cores.cluster <- makePSOCKcluster(6) 
registerDoParallel(cores.cluster)

# 1. Read in data
train_lm <- read_rds('data/finalized_data/final_train_lm.rds') 
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')


# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)

# 3. Define model, recipe, and workflow
prophet_model <- prophet_reg() |> 
  set_engine("prophet", 
             growth = "linear", # linear/logistic 
             changepoint_num = tune(), # num of potential changepoints for trend
             changepoint_range = tune(), # adjusts flexibility of trend
             seasonality_yearly = FALSE, 
             seasonality_weekly = FALSE,
             seasonality_daily = TRUE, # daily data
             season = "additive", # additive/multiplicative
             prior_scale_seasonality = tune(), # strength of seasonality model - larger fits more fluctuations
             prior_scale_holidays = tune(),# strength of holidays component
             prior_scale_changepoints = tune()) 

prophet_recipe <- recipe(new_cases ~ ., data = train_lm) |> 
  step_dummy(all_nominal_predictors()) %>% 
  # trying additional step
  step_normalize(all_numeric_predictors())

# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_wflow <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

# 4. Setup tuning grid

# same parameters for both
prophet_params <- prophet_wflow |> 
  extract_parameter_set_dials()

prophet_grid <- grid_regular(prophet_params, levels = 3)

# 5. Model Tuning 
prophet_multi_tuned <- tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)


stopCluster(cores.cluster)

save(prophet_multi_tuned, file = "Models/cindy/results/prophet_multi_tuned_1.rda")

# 6. Review the best results
# show_best(prophet_multi_tuned, metric = "rmse")
