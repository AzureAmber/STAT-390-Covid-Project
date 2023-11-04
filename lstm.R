library(tidyverse)
library(keras)
library(tensorflow)
library(doParallel)
library(parallel)
library(tictoc)
library(forecast)
library(tseries)


# Source
# http://datasideoflife.com/?p=1171
# https://www.datatechnotes.com/2019/01/regression-example-with-lstm-networks.html


# NOTE: will need to set seed on final run

# Setup parallel processing
#detectCores() # 8
cores.cluster <- makePSOCKcluster(4)
registerDoParallel(cores.cluster)


# 1. Read in data
train_nn = readRDS('data/finalized_data/final_train_nn.rds')
test_nn = readRDS('data/finalized_data/final_test_nn.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds = rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
data_folds

# 3. Rescale input data 

# question: do we need to rescale data for LSTM?

scale_factor <- c(mean(train_nn$new_cases), sd(train_nn$new_cases))

scaled_train_nn <- train_nn %>% 
  select(new_cases) %>% 
  mutate(new_cases = (new_cases - scale_factor[1]) / scale_factor[2])

prediction <- 12
lag <- prediction

scaled_train_nn <-as.matrix(scaled_train_nn)

x_train_data <- t(sapply(
  1:(length(scaled_train_nn) - lag - prediction + 1),
  function(x) scaled_train_nn[x:(x + lag - 1), 1]
))

x_train_arr <- array(
  data = as.numeric(unlist(x_train_data)),
  dim = c(
    nrow(x_train_data),
    lag,
    1
  )
)


# 3. Define model, recipe, and workflow

## 1) how to determine size of the layer
## 2) confirm the batch size, timesteps, and features

lstm_model <- keras_model_sequential() %>% 
  layer_lstm(units = 50, #size of the layer
             batch_input_shape =  c(25166,12, 36), #batch size, timesteps, features,
             return_sequences = TRUE,
             stateful = TRUE) %>% 
  layer_dropout(rate=0.5) %>% 
  layer_lstm(units = 50,
             return_sequences = TRUE,
             stateful = TRUE) %>% 
  layer_dropout(rate = 0.5) %>% 
  time_distributed(layer_dense(units = 1))

lstm_model %>% 
  compile(loss = 'mae', optimizer = 'adam', metrics =  )



########### UNFINISHED, HAVE QUESTIONS TO ASK #####################

# 5. Model Tuning
tic.clearlog()
tic('arima')

arima_tuned = tune_grid(
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





