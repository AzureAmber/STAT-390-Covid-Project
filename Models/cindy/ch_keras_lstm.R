library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)


# Setup parallel processing
cores <- detectCores()
cores.cluster <- makePSOCKcluster(6) 
registerDoParallel(cores.cluster)

# 1. Read in data
train_lm <- read_rds('data/finalized_data/final_train_lm.rds') 
test_lm <- read_rds('data/finalized_data/final_test_lm.rds')
