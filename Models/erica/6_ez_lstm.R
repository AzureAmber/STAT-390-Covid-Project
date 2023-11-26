library(tidyverse)
library(keras)
library(tensorflow)
library(doParallel)
library(parallel)
library(tictoc)
library(forecast)
library(tseries)
use_condaenv("keras-tf", required = T)


# Source
# http://datasideoflife.com/?p=1171
# https://www.datatechnotes.com/2019/01/regression-example-with-lstm-networks.html
# https://www.youtube.com/watch?v=rn8HrlHICfE


# NOTE: will need to set seed on final run

# Setup parallel processing
#detectCores() # 8
cores.cluster <- makePSOCKcluster(4)
registerDoParallel(cores.cluster)


# 1. Read in data
train_nn <- readRDS('data/avg_final_data/final_train_nn.rds')
train_nn <- train_nn %>% 
  group_by(location) %>% 
  arrange(location) %>% 
  ungroup() %>% 
  filter(date > as.Date("2020-01-19"))
write_csv(train_nn, file ="models/erica/train_nn.csv")

test_nn <- readRDS('data/avg_final_data/final_test_nn.rds') 
test_nn <- test_nn %>% 
  group_by(location) %>% 
  arrange(location) %>% 
  ungroup() %>% 
  filter(date > as.Date("2020-01-19"))
write_csv(test_nn, file = "models/erica/test_nn.csv")

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
  compile(loss = 'mae', optimizer = 'adam', metrics = "mse" )





