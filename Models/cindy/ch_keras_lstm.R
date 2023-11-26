library(tidyverse)
library(keras)
library(tensorflow)
library(tidymodels)
library(reticulate)

# Source
# https://cran.r-project.org/web/packages/keras/vignettes/sequential_model.html
# https://tensorflow.rstudio.com/tutorials/keras/regression

# Testing TensorFlow and Keras in R
tf$constant("Hello TensorFlow")
tf$keras$layers$Dense(units = 1)


# 1. Read in data ----
train_nn <- read_rds('data/avg_final_data/final_train_nn.rds') |> 
  filter(date >= as.Date("2020-01-04")) |> 
  mutate(one_day_lag = lag(new_cases, n = 1),
         one_week_lag = lag(new_cases, n = 7),
         one_month_lag = lag(new_cases, n = 30))

test_nn <- read_rds('data/avg_final_data/final_test_nn.rds') |> 
  mutate(one_day_lag = lag(new_cases, n = 1),
         one_week_lag = lag(new_cases, n = 7),
         one_month_lag = lag(new_cases, n = 30))


# 2. Split features from labels (label = target var = new_cases) ----
train_features <- train_nn |> select(where(~is.numeric(.x))) |> select(-new_cases)
test_features <- test_nn|> select(where(~is.numeric(.x))) |> select(-new_cases)

train_labels <- train_nn %>% select(new_cases)
test_labels <- test_nn %>% select(new_cases)

# 3. Normalize ----
# Calculate mean and standard deviation for normalization
train_mean <- apply(train_features, 2, mean, na.rm = TRUE)
train_sd <- apply(train_features, 2, sd, na.rm = TRUE)

# Normalize the training data
train_features_norm <- sweep(train_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")

# Normalize the test data using training mean and sd
test_features_norm <- sweep(test_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")

# 4. Reshape Data for LSTM ----
# Determine the number of time steps and features
time_steps <- 30 
features <- ncol(train_features_norm)

# Function to create 3D array for LSTM
create_dataset <- function(data, time_steps) {
  data <- as.matrix(data)
  X <- array(NA, dim = c(nrow(data) - time_steps, time_steps, ncol(data)))
  for (i in seq_len(nrow(data) - time_steps)) {
    X[i,,] <- data[i:(i + time_steps - 1),]
  }
  return(X)
}

train_data_lstm <- create_dataset(train_features_norm, time_steps)  # Your function to create 3D array
train_labels_lstm <- as.matrix(train_labels_vec[(time_steps + 1):length(train_labels_vec)])

# Convert train_labels and test_labels to vectors if they are dataframes
train_labels_vec <- as.vector(train_labels$new_cases)
test_labels_vec <- as.vector(test_labels$new_cases)

# Adjust the length of train_labels and test_labels to match the reshaped data
train_labels_adj <- train_labels_vec[(time_steps + 1):length(train_labels_vec)]
test_labels_adj <- test_labels_vec[(time_steps + 1):length(test_labels_vec)]


# 4. Linear regression w/ multiple inputs
# lstm_model <- keras_model_sequential() |> # linear stack of layers
#   # layer_lstm(units = 10, input_shape = c(time_steps, features)) |> 
#   # layer_dropout(rate = 0.2) |> 
#   layer_normalizer(axis = -1) |> 
#   # units rep # of neurons in layer
#   layer_dense(units = 1)  # 1 for single output regression
#   # handles sequences; units ~ "memory" of layer

lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(time_steps, features), return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)  # Output layer for regression prediction


# Compile model 
rmse <- function(y_true, y_pred) {
  sqrt(mean((y_true - y_pred)^2))
}

lstm_model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error',
  metrics = c(rmse, 'mean_absolute_error')
)


summary(lstm_model)

# Fit the model to the training data
history <- lstm_model %>% fit(
  train_data_lstm,  
  train_labels_lstm, 
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2  # use a portion of the training data for validation
)

performance <- lstm_model %>% evaluate(
  test_data_lstm,  # your test dataset
  test_labels      # your test labels
)

print(model_performance)

