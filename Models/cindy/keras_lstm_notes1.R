# Load Libraries
library(tidyverse)
library(tidymodels)
library(doParallel)
library(keras)
library(tensorflow)
library(ModelMetrics)
library(reticulate)

# Testing TensorFlow and Keras in R
tf$constant("Hello TensorFlow")
tf$keras$layers$Dense(units = 1)

# 1. Read Data ----
final_train_nn <- read_rds('data/avg_final_data/final_train_nn.rds')
final_test_nn <- read_rds('data/avg_final_data/final_test_nn.rds')

# 2. Preprocess Data ----
resp_scale <- c(mean(final_train_nn$new_cases), sd(final_train_nn$new_cases))

# 3. Combine and Clean Data ----
complete_nn <- final_train_nn %>% 
  rbind(final_test_nn) %>%
  filter(date >= as.Date("2020-01-04")) %>%
  group_by(location) %>%
  select(date, new_cases, location, new_deaths, new_tests, new_tests_b, gdp_per_capita) %>%
  ungroup()

# 4. Encode Country Names ----
country_names <- sort(unique(complete_nn$location))
country_convert <- tibble(location = country_names, locid = 1:length(country_names))
complete_nn <- complete_nn %>% left_join(country_convert, by = "location")

# 5. Normalize Data ----
data_norm <- complete_nn %>%
  mutate_at(vars(new_cases, new_deaths, new_tests, gdp_per_capita), scale) %>%
  mutate(new_tests_b = ifelse(new_tests_b, 1, 0)) %>%
  group_by(location) %>%
  mutate(
    new_cases_lag7 = lag(new_cases, 7, default = NA),
    locid_lag7 = lag(locid, 7, default = NA),
    new_deaths_lag7 = lag(new_deaths, 7, default = NA),
    new_tests_lag7 = lag(new_tests, 7, default = NA),
    new_tests_b_lag7 = lag(new_tests_b, 7, default = NA),
    gdp_lag7 = lag(gdp_per_capita, 7, default = NA)
  ) %>%
  ungroup() %>%
  drop_na()

# 6. Split Data ----
train_nn <- data_norm %>% filter(date < as.Date("2023-01-01"))
test_nn <- data_norm %>% filter(date >= as.Date("2023-01-01"))

# 7. Function to Create 3D Array ----
create_3d_array <- function(data, predictors, response_var) {
  data_x <- array(0, dim = c(nrow(data), 1, length(predictors)))
  for (i in seq_along(predictors)) {
    data_x[,,i] <- as.matrix(data %>% select(!!sym(predictors[i])))
  }
  data_y <- array(data[[response_var]], dim = c(nrow(data), 1, 1))
  list(data_x, data_y)
}

# 8. LSTM Model Training and Prediction ----
train_and_predict <- function(train_data, test_data) {
  # Create 3D Arrays for Training and Testing
  predictors <- c("new_cases_lag7", "locid_lag7", "new_deaths_lag7", "new_tests_lag7", "new_tests_b_lag7", "gdp_lag7")
  list_train <- create_3d_array(train_data, predictors, "new_cases")
  list_test <- create_3d_array(test_data, predictors, "new_cases")
  
  # Model Parameters
  num_units <- 30
  num_epochs <- 10
  
  # Build LSTM Model
  cores.cluster <- makePSOCKcluster(20)
  registerDoParallel(cores.cluster)
  
  lstm_model <- keras_model_sequential() %>%
    layer_lstm(units = num_units, batch_input_shape = c(1, 1, 6), return_sequences = TRUE, stateful = TRUE) %>%
    layer_dropout(rate = 0.5) %>%
    layer_lstm(units = num_units, return_sequences = TRUE, stateful = TRUE) %>%
    layer_dense(units = 1)
  
  lstm_model %>% compile(loss = "mse", optimizer = "adam", metrics = "mae")
  
  # Train Model
  for (i in 1:num_epochs) {
    lstm_model %>% fit(x = list_train[[1]], y = list_train[[2]], batch_size = 1, epochs = 1, verbose = 0, shuffle = FALSE)
    lstm_model %>% reset_states()
  }
  
  stopCluster(cores.cluster)
  
  # Predictions
  fitted_train <- predict(lstm_model, list_train[[1]], batch_size = 1) %>% .[,,1]
  fitted_test <- predict(lstm_model, list_test[[1]], batch_size = 1) %>% .[,,1]
  
  # Rescale Predictions
  fitted_train <- fitted_train * resp_scale[2] + resp_scale[1]
  fitted_test <- fitted_test * resp_scale[2] + resp_scale[1]
  
  list(fitted_train, fitted_test)
}

# 9. Loop Through Each Country ----
cl = makePSOCKcluster(6)
registerDoParallel(cl)

results <- list()
for (country in country_names) {
  cat("Processing", country, "\n")
  
  # Subset Data for Current Country
  train_data <- train_nn %>% filter(location == country)
  test_data <- test_nn %>% filter(location == country)
  
  # Train and Predict
  predictions <- train_and_predict(train_data, test_data)
  
  # Calculate RMSE
  rmse_train <- rmse(train_data$new_cases, predictions[[1]])
  rmse_test <- rmse(test_data$new_cases, predictions[[2]])
  
  # Store Results
  results[[country]] <- list(rmse_train = rmse_train, rmse_test = rmse_test)
  
  # Create Plots
  plot_data_train <- train_data %>% 
    mutate(pred = predictions[[1]]) %>% 
    select(date, new_cases, pred)
  
  plot_data_test <- test_data %>% 
    mutate(pred = predictions[[2]]) %>% 
    select(date, new_cases, pred)
  
  p_train <- ggplot(plot_data_train, aes(date)) +
    geom_line(aes(y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(y = pred, color = "Predicted"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = paste("Training: Actual vs Predicted New Cases in", country),
         caption = "Keras LSTM",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  filename = paste("Results/cindy/lstm_avg/training_plots/lstm_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  ggsave(filename, p_train, width = 10, height = 6)
  
  
  p_test <- ggplot(plot_data_test, aes(date)) +
    geom_line(aes(y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(y = pred, color = "Predicted"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = paste("Testing: Actual vs Predicted New Cases in", country),
         caption = "Keras LSTM",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  filename = paste("Results/cindy/lstm_avg/testing_plots/lstm_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  ggsave(filename, p_test, width = 10, height = 6)

}

stopCluster(cl)
# View Results
results
