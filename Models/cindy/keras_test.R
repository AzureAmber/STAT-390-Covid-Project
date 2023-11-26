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
  drop_na()

test_nn <- read_rds('data/avg_final_data/final_test_nn.rds') %>% 
  filter(date >= as.Date("2020-01-04")) %>%
  mutate(
    one_day_lag = lag(new_cases, n = 1),
    one_week_lag = lag(new_cases, n = 7),
    one_month_lag = lag(new_cases, n = 30)
  ) %>%
  drop_na()

# Split features from labels
train_features <- train_nn %>% select(where(is.numeric)) %>% select(-new_cases)
train_labels <- train_nn %>% pull(new_cases)
test_features <- test_nn %>% select(where(is.numeric)) %>% select(-new_cases)
test_labels <- test_nn %>% pull(new_cases)

# Normalize
train_mean <- apply(train_features, 2, mean, na.rm = TRUE)
train_sd <- apply(train_features, 2, sd, na.rm = TRUE)

train_features_norm <- sweep(train_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")
test_features_norm <- sweep(test_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")

# Reshape Data for LSTM
time_steps <- 30
features <- ncol(train_features_norm)

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

train_labels_adj <- train_labels[(time_steps + 1):length(train_labels)]
test_labels_adj <- test_labels[(time_steps + 1):length(test_labels)]

# Define LSTM Model
lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(time_steps, features), return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

# Custom RMSE Metric Function
rmse <- function(y_true, y_pred) {
  sqrt(mean((y_true - y_pred)^2))
}

# Compile Model
lstm_model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error',
  metrics = list('mean_absolute_error', rmse)
)

# Model Summary
summary(lstm_model)

# Fit the Model
history <- lstm_model %>% fit(
  train_data_lstm,  
  train_labels_adj, 
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate Model Performance
performance <- lstm_model %>% evaluate(
  test_data_lstm,  
  test_labels_adj
)

print(performance)

# Predict on the test data
test_predictions <- lstm_model %>% predict(test_data_lstm)

# Since test_predictions will be a matrix, convert it to a vector if necessary
test_predictions <- test_predictions[, 1]

# If you want to compare predictions with actual values, you can create a data frame
comparison <- tibble(
  Actual = test_labels_adj,
  Predicted = test_predictions
)

# Adding a sequence number to the comparison dataframe
comparison <- comparison %>% 
  mutate(Observation = row_number())

# Visualizing the predictions vs actual values
ggplot(comparison, aes(x = Observation)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "LSTM Model Predictions vs Actual Values", x = "Observation", y = "Value") +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))



