library(tidyverse)
library(keras)
library(tensorflow)
library(tidymodels)
library(reticulate)

# Read in data
train_nn <- read_rds('data/avg_final_data/final_train_nn.rds') %>% 
  filter(date >= as.Date("2020-01-04")) %>% 
  mutate(
    one_day_lag = lag(new_cases, n = 1),
    one_week_lag = lag(new_cases, n = 7),
    one_month_lag = lag(new_cases, n = 30)
  ) %>%
  drop_na()  # Drop NA values resulting from lagging

test_nn <- read_rds('data/avg_final_data/final_test_nn.rds') %>% 
  mutate(
    one_day_lag = lag(new_cases, n = 1),
    one_week_lag = lag(new_cases, n = 7),
    one_month_lag = lag(new_cases, n = 30)
  ) %>%
  drop_na()  # Drop NA values resulting from lagging

# Split features from labels
train_features <- train_nn %>% select(where(is.numeric)) %>% select(-new_cases)
train_labels <- train_nn %>% pull(new_cases)  # Convert to vector
test_features <- test_nn %>% select(where(is.numeric)) %>% select(-new_cases)
test_labels <- test_nn %>% pull(new_cases)  # Convert to vector

# Normalize
train_mean <- apply(train_features, 2, mean, na.rm = TRUE)
train_sd <- apply(train_features, 2, sd, na.rm = TRUE)

train_features_norm <- sweep(train_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")
test_features_norm <- sweep(test_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")

# Reshape Data for LSTM
time_steps <- 30  # Number of time steps in LSTM
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

train_data_lstm <- create_dataset(train_features_norm, time_steps)
test_data_lstm <- create_dataset(test_features_norm, time_steps)

# Adjust the length of train_labels and test_labels to match the reshaped data
train_labels_adj <- train_labels[(time_steps + 1):length(train_labels)]
test_labels_adj <- test_labels[(time_steps + 1):length(test_labels)]

# Define LSTM Model
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
  metrics = list('mean_absolute_error', rmse)  # Use the built-in MAE and custom RMSE
)

summary(lstm_model)

# Fit the model to the training data
history <- lstm_model %>% fit(
  train_data_lstm,  
  train_labels_adj, 
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate model performance on the test data
performance <- lstm_model %>% evaluate(
  test_data_lstm,  
  test_labels_adj
)

print(performance)
      