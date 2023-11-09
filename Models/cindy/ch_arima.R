library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(tictoc)

tidymodels_prefer()

# Source
# https://www.rdocumentation.org/packages/modeltime/versions/1.2.8/topics/arima_reg
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/



# Setup parallel processing
# detectCores() # 8
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
data_folds

# 3. Define model, recipe, and workflow
arima_model <- arima_reg(
  seasonal_period = "auto", # default
  non_seasonal_ar = tune(), # p (0-5)
  non_seasonal_differences = tune(), # d (0-2)
  non_seasonal_ma = tune(), # q (0-5)
  # values below based on example
  seasonal_ar = 1, 
  seasonal_differences = 0, 
  seasonal_ma = 1
  
) |> 
  set_engine("arima")


arima_recipe <- recipe(new_cases ~ date, data = train_lm) 

arima_wflow <- workflow() %>%
  add_model(arima_model) %>%
  add_recipe(arima_recipe)

# 4. Setup tuning grid
arima_params <- arima_wflow |> 
  extract_parameter_set_dials() |> 
  update(non_seasonal_ar = non_seasonal_ar(c(0, 5)),
         non_seasonal_ma = non_seasonal_ma(c(0, 5)))

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

save(arima_tuned, arima_tictoc, file = "Models/cindy/results/arima_tuned_1.rda")

# 6. Results
show_best(btree_tuned, metric = "rmse")

# non_seasonal_ar non_seasonal_differences non_seasonal_ma .metric .estimator   mean     n std_err
# <int>                    <int>           <int> <chr>   <chr>       <dbl> <int>   <dbl>
#              5                        0               5 rmse    standard   41737.   201   3660.
#              5                        0               2 rmse    standard   43099.   205   3734.
#              5                        1               5 rmse    standard   43099.   201   3717.
#              2                        1               5 rmse    standard   43122.   205   3704.
#              2                        0               2 rmse    standard   43171.   205   3710.

# NOTE: non_seasonal_ar = 5, non_seasonal_differences = 0, non_seasonal_ma = 5
autoplot(arima_tuned, metric = "rmse")

# 7. Fit best model
arima_model_new <- arima_reg(
  seasonal_period = "auto", # default
  non_seasonal_ar = 5, # from above
  non_seasonal_differences = 0, # from above
  non_seasonal_ma = 5, # from above
  # values below based on example
  seasonal_ar = 1, 
  seasonal_differences = 0, 
  seasonal_ma = 1
) |> 
  set_engine("arima")

arima_wflow_new <- workflow() %>%
  add_model(arima_model_new) %>%
  add_recipe(arima_recipe)

arima_fit <- fit(arima_wflow_new, data = train_lm)

final_arima_train <- train_lm %>%
  bind_cols(predict(arima_fit, new_data = train_lm)) %>%
  rename(pred = .pred) 


