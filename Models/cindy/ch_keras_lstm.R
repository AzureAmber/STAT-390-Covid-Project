# Installing TensorFlow packages
# install.packages("remotes")
remotes::install_github("rstudio/tensorflow", force = TRUE)
reticulate::install_python()
# install.packages("keras")
library(tensorflow)
install_tensorflow(version = "nightly")  
library(keras)
# install_keras()
library(reticulate)
reticulate::py_config()

library(tidyverse)
library(tidymodels)
library(doParallel)


# Source
# https://cran.r-project.org/web/packages/keras/vignettes/sequential_model.html
# https://tensorflow.rstudio.com/tutorials/keras/regression
# https://tensorflow.rstudio.com/install/


# Testing TensorFlow and Keras in R
tf$constant("Hello TensorFlow")
# Should return: tf.Tensor(b'Hello TensorFlow', shape=(), dtype=string)

# 1. Read in data ----
train_nn <- read_rds('data/avg_final_data/final_train_nn.rds') 

test_nn <- read_rds('data/avg_final_data/final_test_nn.rds') 

# Decreasing number of predictors ----
cor_matrix <- cor(train_nn[, sapply(train_nn, is.numeric)])

cor_matrix |> 
  as_tibble(rownames = "variable") |> 
  pivot_longer(cols = -variable, names_to = "compared_with", values_to = "correlation") |> 
  filter(variable == "new_cases") |> 
  arrange(desc(correlation))

# variable  compared_with           correlation
# <chr>     <chr>                         <dbl>
# new_cases new_cases                     1    
# new_cases new_deaths                    0.490
# new_cases new_tests                     0.474
# new_cases total_cases                   0.429
# new_cases total_tests                   0.426
# new_cases total_deaths                  0.379
# new_cases new_cases_per_million         0.367
# new_cases total_cases_per_million       0.311
# new_cases gdp_per_capita                0.238
# new_cases female_smokers                0.235

train_nn <- train_nn |> 
  filter(date < as.Date("2023-01-01")) |> 
  group_by(location) |>
  mutate(one_week_lag = lag(new_cases, n = 7),
         one_month_lag = lag(new_cases, n = 30)) |> 
  select(date, new_cases, location, new_deaths, new_tests, gdp_per_capita, female_smokers, contains("lag")) |> 
  ungroup() |> 
  drop_na()

test_nn <- test_nn |> 
  filter(date >= as.Date("2023-01-01")) |> 
  group_by(location) |> 
  mutate(one_week_lag = lag(new_cases, n = 7),
         one_month_lag = lag(new_cases, n = 30)) |> 
  select(date, new_cases, location, new_deaths, new_tests, gdp_per_capita, female_smokers, contains("lag")) |> 
  ungroup() |> 
  drop_na()
  

# 2. Split features from labels (label = target var = new_cases) ----
train_features <- train_nn |> select(where(~is.numeric(.x))) |> select(-new_cases)
test_features <- test_nn |> select(where(~is.numeric(.x))) |> select(-new_cases)

train_labels <- train_nn |> select(new_cases)
test_labels <- test_nn |> select(new_cases)

# 3. Normalize ----
# Calculate mean and standard deviation for normalization
train_mean <- apply(train_features, 2, mean, na.rm = TRUE)
train_sd <- apply(train_features, 2, sd, na.rm = TRUE)

# > train_mean
# new_deaths      new_tests gdp_per_capita female_smokers    one_day_lag   one_week_lag  one_month_lag 
# 154.51502   194302.74150    25562.08774       10.90435    16661.20618    16651.80754    16606.70425 
# > train_sd
# new_deaths      new_tests gdp_per_capita female_smokers    one_day_lag   one_week_lag  one_month_lag 
# 4.082571e+02   4.009710e+05   1.625928e+04   8.907208e+00   4.499279e+04   4.499405e+04   4.498953e+04 

# Normalize the training data
train_features_norm <- sweep(train_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")

# Normalize the test data using training mean and sd
test_features_norm <- sweep(test_features, 2, train_mean, FUN = "-") %>%
  sweep(2, train_sd, FUN = "/")

# 3. Reshape data ----
# Training Predictors
num_features <- ncol(train_features_norm)  # Number of features = 6
data_train_x <- array(0, dim = c(nrow(train_nn), 1, num_features))
for (i in 1:num_features) {
  data_train_x[,,i] <- as.matrix(train_nn %>% select(names(train_features_norm)[i]))
}
dim(data_train_x) # 24476     1     1

# Training Response
data_train_y <- array(0, dim = c(nrow(train_nn), 1, 1))
data_train_y[,,1] <- train_nn$new_cases
dim(data_train_y) # 24476     1     1

# Testing Predictors
data_test_x <- array(0, dim = c(nrow(test_nn), 1, num_features))
for (i in 1:num_features) {
  data_test_x[,,i] <- as.matrix(test_nn %>% select(names(test_features_norm)[i]))
}
dim(data_test_x) # 5210    1    1

# Testing Response
data_test_y <- array(0, dim = c(nrow(test_nn), 1, 1))
data_test_y[,,1] <- test_nn$new_cases
dim(data_test_y) # 5210    1    1

# 5. Linear regression w/ multiple inputs ----
lstm_model <- keras_model_sequential() # linear stack of layers

lstm_model |> 
  # batch_input_shape (# samples, # timesteps, # features per timestep)
  layer_lstm(units = 32, batch_input_shape = c(1, 1, 6), return_sequences = TRUE, stateful = TRUE) |>
  layer_dropout(rate = 0.2) |>
  # units rep # of neurons in layer
  layer_lstm(units = 32, return_sequences = TRUE, stateful = TRUE) |> 
  layer_dense(units = 1)  # 1 for single output regression (# Output layer for regression prediction)
  # handles sequences; units ~ "memory" of layer


# Compile the model
# Adjust the optimizer with a lower learning rate and gradient clipping
optimizer <- optimizer_adam(lr = 0.001, clipnorm = 1)

# Compile the model
lstm_model %>% compile(
  optimizer = optimizer,
  loss = 'mean_squared_error',
  metrics = list('mean_absolute_error')
)

summary(lstm_model)

num_epochs <- 20  # Adjust the number of epochs here

for (i in 1:num_epochs) {
  lstm_model |>  fit(
    data_train_x,  
    data_train_y, 
    batch_size = 1,
    epochs = 1,  # Each iteration of the loop is one epoch
    shuffle = FALSE
  )
  lstm_model %>% reset_states()
}

# Evaluate the model on test data
# Define the inverse normalization function
inverse_normalize <- function(x, mean, sd) {
  (x * sd) + mean
}

mean_new_cases <- train_nn$new_cases |> mean()
sd_new_cases <- train_nn$new_cases |> sd()

train_fit <- predict(lstm_model, data_train_x, batch_size = 1) 
final_train <- train_nn |> 
  mutate(pred = train_fit,
         pred = inverse_normalize(pred, mean_new_cases, sd_new_cases))
train_results <- final_train |> 
  group_by(location) |> 
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)


test_fit <- predict(lstm_model, data_test_x, batch_size = 1) 
final_test <- test_nn |> 
  mutate(pred = test_fit,
         pred = inverse_normalize(pred, mean_new_cases, sd_new_cases))

test_results <- final_test %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)


# Training plots
final_train |> 
  filter(location == "Ethiopia") |> 
  ggplot(aes(x = date)) + 
  geom_line(aes(y = new_cases, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(title = "Training: Actual vs. Predicted New Cases in",
       subtitle = "",,
       caption = "Keras LSTM",
       color = "",
       x = "Date",
       y = "New Cases",
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))


## Training Actual vs. Pred plots ----
# Loop through each country
for(country in unique(train_nn$location)) {
  # Create plot for the current country
  plot_name <- paste("Training: Actual vs. Predicted New Cases in", country)
  file_name <- paste("Results/cindy/lstm_avg/training_plots/lstm_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  train_plot <- ggplot(final_train |> filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         caption = "Keras LSTM",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, train_plot, width = 10, height = 6)
}


## Testing Actual vs. Pred plots ----
# Loop through each country
for(country in unique(train_nn$location)) {
  # Create plot for the current country
  plot_name <- paste("Testing: Actual vs. Predicted New Cases in", country)
  file_name <- paste("Results/cindy/lstm_avg/testing_plots/lstm_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  test_plot <- ggplot(final_test |> filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() +
    labs(x = "Date",
         y = "New Cases",
         title = plot_name,
         caption = "Keras LSTM",
         color = "") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, test_plot, width = 10, height = 6)
}


# save(final_train, final_test, train_results, test_results, file = "Models/cindy/results/lstm_avg.rda")

