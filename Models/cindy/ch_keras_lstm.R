library(tidyverse)
library(tidymodels)
library(doParallel)
library(keras)
library(tensorflow)

# Source
# https://cran.r-project.org/web/packages/keras/vignettes/sequential_model.html
# https://tensorflow.rstudio.com/tutorials/keras/regression

# Setup parallel processing
# cores <- detectCores()
cores.cluster <- makePSOCKcluster(6) 
registerDoParallel(cores.cluster)

# 1. Read in data
train_nn <- read_rds('data/finalized_data/final_train_nn.rds') 
test_nn <- read_rds('data/finalized_data/final_test_nn.rds')

# train_nn |> 
#   GGally::ggpairs(cardinality_threshold = 23)

# 2. Split features from labels (label = target var = new_cases)
train_features <- train_nn |> select(where(~is.numeric(.x))) |> select(-new_cases)
test_features <- test_nn|> select(where(~is.numeric(.x))) |> select(-new_cases)

train_labels <- train_nn %>% select(new_cases)
test_labels <- test_nn %>% select(new_cases)

# 3. Normalize (NEED TO FIX FROM HERE)
normalizer <- layer_normalization(axis = -1L)
normalizer |> adapt(as.matrix(train_features))

lstm_model <- keras_model_sequential() |> # linear stack of layers
  # more complex data requires more layers 
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(timesteps, n_features)) |> 
  layer_dropout(rate = 0.2) |> 
  layer_lstm(units = 50) |> 
  layer_dropout(rate = 0.2) |> 
  layer_dense(units = output_units)

# Compile model 
lstm_model |> compile(
  optimizer = "adam", # follows adam alg
  loss = "mean_squared_error", # want to minimize
  metrics = c("rmse")
)

history <- lstm_model |> fit(
  train_nn$date, train_nn$new_cases, 
  epochs = 100, 
  batch_size = 32
)
