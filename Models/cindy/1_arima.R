library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)
library(forecast)
library(tseries)

tidymodels_prefer()

# Source
# https://business-science.github.io/modeltime/reference/arima_reg.html 
# https://www.youtube.com/watch?v=zB_0Yxxs0b4 

# NOTE: will need to set seed on final run

# Setup parallel processing
detectCores() # 8
cores.cluster <- makePSOCKcluster(4)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm <- readRDS('data/finalized_data/final_train_lm.rds')
test_lm <- readRDS('data/finalized_data/final_test_lm.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
data_folds

# 3. Define model, recipe, and workflow
arima_model <- arima_reg(
  # auto by default
  seasonal_period = "auto",
  non_seasonal_ar = tune(),
  non_seasonal_differences = tune(),
  non_seasonal_ma = tune(),
  # used values from example
  seasonal_ar              = 1,
  seasonal_differences     = 0,
  seasonal_ma              = 1
) %>%
  set_engine("arima") 

arima_recipe <- recipe(new_cases ~ ., data = train_lm) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

arima_wflow <- workflow() %>%
  add_model(arima_model) %>%
  add_recipe(arima_recipe)

# 4. Setup tuning grid
arima_params <- arima_wflow %>%
  extract_parameter_set_dials() 

arima_grid <- grid_regular(arima_params, levels = 3)

# 5. Model Tuning
tic.clearlog()
tic('arima')

arima_tuned <- tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

toc(log = TRUE)
time_log <- tic.log(format = FALSE)
arima_tictoc <- tibble(model = time_log[[1]]$msg, 
                            runtime = time_log[[1]]$toc - time_log[[1]]$tic)
stopCluster(cores.cluster)

# 6. Save results
arima_tuned %>% collect_metrics()


save(arima_tuned, arima_tictoc, file = "Models/cindy/results/arima_tuned.rda")





