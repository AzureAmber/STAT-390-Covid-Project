library(tidyverse)
library(tidymodels)
library(doParallel)
library(tictoc)
library(forecast)
library(tseries)
library(modeltime)
library(dials)
library(timetk)

tidymodels_prefer()

# Setup parallel processing
cores <- detectCores()
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
# 
# # Testing another method
# data_folds2 <- time_series_cv(
#   data = train_lm,
#   initial = 366,
#   assess = 30 * 2,
#   skip = 30 * 4,
#   cumulative = FALSE
# )

# 3. Define model, recipe, and workflow
autoarima_model <- arima_reg() %>% 
  set_engine("auto_arima") %>% 
  set_mode("regression")

autoarima_recipe <- recipe(new_cases ~ ., data = train_lm) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = as.integer(G20),
    G24 = as.integer(G24)
  ) %>%
  step_dummy(all_nominal(), -all_outcomes())

autoarima_wflow <- workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

# 4. Fit the model to the rolling origin resamples
tic("auto arima fitting")

autoarima_fits <- fit_resamples(
  autoarima_wflow,
  resamples = data_folds,
  control = control_resamples(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE)
)

toc(log = TRUE)
time_log <- tic.log(format = FALSE)
autoarima_tictoc <- tibble(model = time_log[[1]]$msg, runtime = time_log[[1]]$toc - time_log[[1]]$tic)

stopCluster(cores.cluster)

# 5. Save results
save(autoarima_fits, autoarima_tictoc, file = "Models/cindy/results/autoarima_results1.rda")

#########################################
# 6. Looking at model results

autoarima_fits |> 
  collect_metrics()
  

