library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(forecast)
library(lubridate)


# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/
# https://www.youtube.com/watch?v=8na-sasmu5I



# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm <- read_rds('./data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('./data/finalized_data/final_test_lm.rds')

# 2. filter out the two countries we will be use

## stationary country rep
train_us <- train_lm %>% 
  filter(location == "United States")
test_us <- test_lm %>% 
  filter(location == "United States")

## non-stationary country rep
train_germ <- train_lm %>% 
  filter(location == "Germany")
test_germ <- test_lm %>% 
  filter(location == "Germany")

# 3. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_us,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds


# # 4. automatic grid search for best p,d,q
# 
# # us: 3,1,5
# train_us %>% 
#   select(new_cases) %>% 
#   ts() %>% 
#   auto.arima() %>% 
#   summary()
# 
# # germany:2,1,2
# train_germ %>% 
#   select(new_cases) %>% 
#   ts() %>% 
#   auto.arima() %>% 
#   summary()

# # 5. manual grid search for best p,d,q
# 
# p_range <- 0:5
# d_range <- 0:2
# q_range <- 0:5
# 
# best_aic <- Inf
# best_order <- c(0,0,0)
# best_model <- NULL
# 
# for (p in p_range){
#   for(d in d_range){
#     for(q in q_range){
#       model <- tryCatch(Arima(train_germ, order = c(p,d,q)),
#                         error = function(e) NULL)
#       
#       if(!is.null(model)){
#         if (model$aic < best_aic){
#           best_aic <- model$aic
#           best_order <- c(p, d, q)
#           best_model <- model
#         }
#       }
#     }
#   }
# }
# 
# 
# best_order
# best_aic
# 
# forecast <- forecast(best_model, h = 12)
# 
# plot(forecast)
# 
# # 6. compare manual search and automatic search results by AIC
# 
# manual_model_us <- Arima(train_us$new_cases, order = c(0,0,0))
# print(AIC(manual_model_us)) #28834.8
# 
# auto_model_us <- auto.arima(train_us$new_cases)
# print(AIC(auto_model)) #26758.13
# 
# 
# manual_model_germ <- Arima(train_germ$new_cases, order = c(0,0,0))
# print(AIC(manual_model_germ)) #29248.58
# 
# auto_model_germ <- auto.arima(train_germ$new_cases)
# print(AIC(auto_model_germ)) #28719.69

# 7. train the model using automatic search result (smaller AIC value)

arima_us <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 3, non_seasonal_differences = 1, non_seasonal_ma = 5,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe <- recipe(new_cases ~ date, data = train_us)

# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

arima_wflow <- workflow() %>%
  add_model(arima_us) %>%
  add_recipe(arima_recipe)

# 4. Setup tuning grid
arima_params <- arima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = seasonal_ar(c(0, 2)),
    seasonal_ma = seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )
arima_grid = grid_regular(arima_params, levels = 3)

# 5. Model Tuning
arima_tuned <- tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

stopCluster(cores.cluster)

arima_tuned %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

save(arima_tuned, file = "Models/erica/results/arima_tuned_us_1.rda")

# 6. Results
arima_us_autoplot <- autoplot(arima_tuned, metric = "rmse")

save(arima_us_autoplot, file = "Models/erica/results/arima_us_autoplot.jpeg")

show_best(arima_tuned, metric = "rmse") #with RMSE: 63021

## BEST MODEL HYPERPARAMTER: p,d,q: 3,1,5; P,D,Q: 2,0,1

# 7. fit test

