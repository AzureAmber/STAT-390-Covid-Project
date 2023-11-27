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

### THE REST ARE DONE IN PYTHON ###

