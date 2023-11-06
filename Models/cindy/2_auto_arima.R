library(tidyverse)
library(tidymodels)
library(forecast)
library(tseries)
library(modeltime)
library(doParallel)

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

# 3. Define model, recipe, and workflow
autoarima_model <- arima_reg() %>% 
  set_engine("auto_arima") 

autoarima_recipe <- recipe(new_cases ~ date, data = train_lm) 

autoarima_wflow <- workflow() %>%
  add_model(autoarima_model) %>%
  add_recipe(autoarima_recipe)

# 4. Fit the model to the rolling origin resamples
autoarima_fits <- fit_resamples(
  autoarima_wflow,
  resamples = data_folds,
  control = control_resamples(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE)
)


autoarima_fit

stopCluster(cores.cluster)

# 5. Results
autoarima_fits |> 
  collect_metrics()

# .metric .estimator       mean     n    std_err .config             
# 1 rmse    standard   43438.       205 3739.      Preprocessor1_Model1
# 2 rsq     standard       0.0170   187    0.00314 Preprocessor1_Model1
