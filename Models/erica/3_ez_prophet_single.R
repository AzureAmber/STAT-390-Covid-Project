library(tidyverse)
library(prophet)
library(modeltime)
library(doParallel)

# Source
# https://www.youtube.com/watch?v=OIQPIefDxx0


##### just throw in original data, no need to preprocessing


# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm <- readRDS('./data/finalized_data/final_train_lm.rds')
test_lm <- readRDS('./data/finalized_data/final_test_lm.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_lm,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds