library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
# library(prophet)


# Source
# https://rdrr.io/cran/modeltime/man/prophet_reg.html
# https://www.youtube.com/watch?v=kyPg3jV4pJ8 


# Setup parallel processing
cores <- detectCores()
cores.cluster <- makePSOCKcluster(6) 
registerDoParallel(cores.cluster)

# 1. Read in data ----
train_lm <- read_rds('data/finalized_data/final_train_lm.rds') 
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')


# 2. Create validation sets ----
# for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)

# 3. Define model, recipe, and workflow ----
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

prophet_recipe <- recipe(new_cases ~ date + location, data = train_lm) 
# NOTE TO SELF: needed to include location in recipe as well
#               since we are predicting on country level

# View(prophet_recipe %>% prep() %>% bake(new_data = NULL))

prophet_wflow <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

# 4. Setup tuning grid ----

# same parameters for both
prophet_params <- prophet_wflow |> 
  extract_parameter_set_dials()

prophet_grid <- grid_regular(prophet_params, levels = 3) # 243 combos

# 5. Model Tuning ----
prophet_tuned <- tune_grid(
  prophet_wflow,
  resamples = data_folds,
  grid = prophet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

save(prophet_tuned, file = "Models/cindy/results/prophet_tuned_1.rda")

# 6. Review the best results ----
show_best(prophet_tuned, metric = "rmse")

# changepoint_num changepoint_range prior_scale_seasonal…¹ prior_scale_holidays prior_scale_changepo…² .metric .estimator   mean     n
# <int>             <dbl>                  <dbl>                <dbl>                  <dbl> <chr>   <chr>       <dbl> <int>
# 1               0              0.6                     100                0.001                  0.316 rmse    standard   29901.   205
# 2               0              0.75                    100                0.001                  0.316 rmse    standard   29901.   205
# 3               0              0.9                     100                0.001                  0.316 rmse    standard   29901.   205
# 4               0              0.6                     100                0.316                  0.316 rmse    standard   29901.   205
# 5               0              0.75                    100                0.316                  0.316 rmse    standard   29901.   205


# NOTE: changepoint_num = 0, changepoint_range = 0.6, prior_scale_changepoint = 100
#       prior_scale_holidays = 0.001, prior_scale_seasonality = 0.316

autoplot(prophet_tuned, metric = "rmse")

# 7. Finalize workflow and fit ----
prophet_wflow_final <- prophet_wflow %>% 
  finalize_workflow(select_best(prophet_tuned, metric = "rmse"))

prophet_fit <- fit(prophet_wflow_final, data = train_lm)

prophet_pred <- predict(prophet_fit, new_data = train_lm) |>  
  bind_cols(train_lm) |> 
  rename(pred = .pred)

ggplot(prophet_pred) + 
  geom_line(aes(x = date, y = new_cases)) + 
  geom_line(aes(x = date, y = pred), color = "red") + 
  facet_wrap(~location) + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Prophet (Single)")

prophet_result <- prophet_pred |> 
  group_by(location) |> 
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |> 
  arrange(location)

# x <- prophet_result |> 
#   pivot_wider(names_from = location, values_from = value)
# view(x)  
