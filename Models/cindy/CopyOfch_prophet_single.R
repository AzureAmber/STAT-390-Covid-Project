library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(prophet)


# Source
# https://rdrr.io/cran/modeltime/man/prophet_reg.html
# https://www.youtube.com/watch?v=kyPg3jV4pJ8 


# Setup parallel processing
cores <- detectCores()
cores.cluster <- makePSOCKcluster(6) 
registerDoParallel(cores.cluster)

# 1. Read in data
train_lm <- read_rds('data/finalized_data/final_train_lm.rds') 
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')
# 
# # NOTE: Using United States (Stationary) & Japan (Non-Stationary)
# train_lm_us <- train_lm |> 
#   filter(location == "United States")
# 
# train_lm_jp <- train_lm |> 
#   filter(location == "Japan")

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
             prior_scale_changepoints = tune(), # strength of seasonality model - larger fits more fluctuations
             prior_scale_holidays = tune()) |> # strength of holidays component
  set_mode("regression")

prophet_recipe <- recipe(new_cases ~ date, data = train_lm)
# prophet_recipe_us <- recipe(new_cases ~ date, data = train_lm_us)
# prophet_recipe_jp <- recipe(new_cases ~ date, data = train_lm_jp)

prophet_wflow <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

# prophet_wflow_us <- workflow() %>%
#   add_model(prophet_model) %>%
#   add_recipe(prophet_recipe_us)
# 
# prophet_wflow_jp <- workflow() %>%
#   add_model(prophet_model) %>%
#   add_recipe(prophet_recipe_jp)

# 4. Setup tuning grid

# same parameters for both
prophet_params <- prophet_wflow |> 
  extract_parameter_set_dials()

prophet_grid <- grid_regular(prophet_params, levels = 3)

# 5. Model Tuning 
prophet_tuned <- tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)
# 
# prophet_tuned_us <- tune_grid(
#   prophet_wflow_us,
#   resamples = data_folds,
#   grid = prophet_grid,
#   control = control_grid(save_pred = TRUE,
#                          save_workflow = FALSE,
#                          parallel_over = "everything"),
#   metrics = metric_set(rmse)
# )
# 
# prophet_tuned_jp <- tune_grid(
#   prophet_wflow_jp,
#   resamples = data_folds,
#   grid = prophet_grid,
#   control = control_grid(save_pred = TRUE,
#                          save_workflow = FALSE,
#                          parallel_over = "everything"),
#   metrics = metric_set(rmse)
# )

stopCluster(cores.cluster)

save(prophet_tuned, file = "Models/cindy/results/prophet_tuned_1.rda")

# save(prophet_tuned_us, prophet_tuned_jp, file = "Models/cindy/results/prophet_tuned_1.rda")

# 6. Review the best results
# show_best(prophet_tuned_us, metric = "rmse")
# show_best(prophet_tuned_jp, metric = "rmse")
  
  
  
  
  
