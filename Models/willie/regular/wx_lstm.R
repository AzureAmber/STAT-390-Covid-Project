library(tidyverse)
library(tidymodels)
library(doParallel)
library(keras)
library(tensorflow)


# Source
# http://rwanjohi.rbind.io/2018/04/05/time-series-forecasting-using-lstm-in-r/




# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)


# 1. Read in data
train_nn = readRDS('data/finalized_data/final_train_nn.rds')
test_nn = readRDS('data/finalized_data/final_test_nn.rds')












