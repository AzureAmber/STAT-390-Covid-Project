library(tidyverse)
library(tidymodels)
library(doParallel)
library(keras)
library(tensorflow)
# install_keras()
# install_tensorflow(version = "nightly")


# Install
# https://tensorflow.rstudio.com/install/

# Sources
# http://datasideoflife.com/?p=1171
# https://algoritmaonline.com/time-series-prediction-with-lstm/
# http://rwanjohi.rbind.io/2018/04/05/time-series-forecasting-using-lstm-in-r/
# https://smltar.com/dllstm
# https://www.r-bloggers.com/2021/04/lstm-network-in-r/

# 1. Read in data
final_train_nn = readRDS('data/finalized_data/final_train_nn.rds')
final_test_nn = readRDS('data/finalized_data/final_test_nn.rds')

# Remove observations before first appearance of COVID: 2020-01-04
# choose predictors and normalize data
# lag 7 days
complete_nn = final_train_nn %>% rbind(final_test_nn) %>%
  filter(date >= as.Date("2020-01-04")) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()

country_num = tibble(
  country = unique(complete_nn$location),
  num = 1:length(unique(complete_nn$location))
)
complete_nn = complete_nn %>%
  left_join(country_num, by = join_by(location == country)) %>%
  mutate(location = num) %>%
  select(-c(num))

data_norm = complete_nn %>%
  select(c(date, location, new_cases, new_deaths, new_tests, gdp_per_capita, new_tests_b)) %>%
  group_by(location) %>%
  mutate(
    new_cases = as.vector(scale(new_cases)),
    new_deaths = as.vector(scale(new_deaths)),
    new_tests = as.vector(scale(new_tests)),
    new_tests_b = ifelse(new_tests_b, 1, 0)
  ) %>%
  mutate(
    # new_cases
    new_cases_lag1 = lag(new_cases, 1, default = NA),
    new_cases_lag2 = lag(new_cases, 2, default = NA),
    new_cases_lag3 = lag(new_cases, 3, default = NA),
    new_cases_lag4 = lag(new_cases, 4, default = NA),
    new_cases_lag5 = lag(new_cases, 5, default = NA),
    new_cases_lag6 = lag(new_cases, 6, default = NA),
    # new_deaths
    new_deaths_lag1 = lag(new_deaths, 1, default = NA),
    new_deaths_lag2 = lag(new_deaths, 2, default = NA),
    new_deaths_lag3 = lag(new_deaths, 3, default = NA),
    new_deaths_lag4 = lag(new_deaths, 4, default = NA),
    new_deaths_lag5 = lag(new_deaths, 5, default = NA),
    new_deaths_lag6 = lag(new_deaths, 6, default = NA),
    # new_tests
    new_tests_lag1 = lag(new_tests, 1, default = NA),
    new_tests_lag2 = lag(new_tests, 2, default = NA),
    new_tests_lag3 = lag(new_tests, 3, default = NA),
    new_tests_lag4 = lag(new_tests, 4, default = NA),
    new_tests_lag5 = lag(new_tests, 5, default = NA),
    new_tests_lag6 = lag(new_tests, 6, default = NA),
    # new_tests_b
    new_tests_b_lag1 = lag(new_tests_b, 1, default = NA),
    new_tests_b_lag2 = lag(new_tests_b, 2, default = NA),
    new_tests_b_lag3 = lag(new_tests_b, 3, default = NA),
    new_tests_b_lag4 = lag(new_tests_b, 4, default = NA),
    new_tests_b_lag5 = lag(new_tests_b, 5, default = NA),
    new_tests_b_lag6 = lag(new_tests_b, 6, default = NA)
  ) %>%
  ungroup() %>%
  drop_na()
train_nn = data_norm %>% filter(date < as.Date("2023-01-01"))
test_nn = data_norm %>% filter(date >= as.Date("2023-01-01"))



# transform data into 3d array
# training predictors
data_train_x = array(0, dim = c(nrow(train_nn), 6, 6))
data_train_x[,,1] = train_nn$location
data_train_x[,,2] = as.matrix(
  train_nn %>%
    select(new_cases_lag1, new_cases_lag2, new_cases_lag3,
           new_cases_lag4, new_cases_lag5, new_cases_lag6)
)
data_train_x[,,3] = as.matrix(
  train_nn %>%
    select(new_deaths_lag1, new_deaths_lag2, new_deaths_lag3,
           new_deaths_lag4, new_deaths_lag5, new_deaths_lag6)
)
data_train_x[,,4] = as.matrix(
  train_nn %>%
    select(new_tests_lag1, new_tests_lag2, new_tests_lag3,
           new_tests_lag4, new_tests_lag5, new_tests_lag6)
)
data_train_x[,,5] = train_nn$gdp_per_capita
data_train_x[,,6] = as.matrix(
  train_nn %>%
    select(new_tests_b_lag1, new_tests_b_lag2, new_tests_b_lag3,
           new_tests_b_lag4, new_tests_b_lag5, new_tests_b_lag6)
)
# training response
data_train_y = array(0, dim = c(nrow(train_nn), 1, 1))
data_train_y[,,1] = train_nn$new_cases
# testing predictors
data_test_x = array(0, dim = c(nrow(test_nn), 6, 6))
data_test_x[,,1] = test_nn$location
data_test_x[,,2] = as.matrix(
  test_nn %>%
    select(new_cases_lag1, new_cases_lag2, new_cases_lag3,
           new_cases_lag4, new_cases_lag5, new_cases_lag6)
)
data_test_x[,,3] = as.matrix(
  test_nn %>%
    select(new_deaths_lag1, new_deaths_lag2, new_deaths_lag3,
           new_deaths_lag4, new_deaths_lag5, new_deaths_lag6)
)
data_test_x[,,4] = as.matrix(
  test_nn %>%
    select(new_tests_lag1, new_tests_lag2, new_tests_lag3,
           new_tests_lag4, new_tests_lag5, new_tests_lag6)
)
data_test_x[,,5] = test_nn$gdp_per_capita
data_test_x[,,6] = as.matrix(
  test_nn %>%
    select(new_tests_b_lag1, new_tests_b_lag2, new_tests_b_lag3,
           new_tests_b_lag4, new_tests_b_lag5, new_tests_b_lag6)
)
# testing response
data_test_y = array(0, dim = c(nrow(test_nn), 1, 1))
data_test_y[,,1] = test_nn$new_cases





# LSTM Model
# build
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)

lstm_model = keras_model_sequential()
lstm_model %>%
  layer_lstm(
    units = 50, batch_input_shape = c(1, 6, 6), return_sequences = TRUE, stateful = TRUE
  ) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(
    units = 50, return_sequences = TRUE, stateful = TRUE
  ) %>%
  layer_dense(units = 1)
# compile
lstm_model %>% compile(loss = "mse", optimizer = "adam", metrics = "mae")
summary(lstm_model)

num_epochs = 20
for (i in 1:num_epochs) {
  lstm_model %>% fit(
    x = data_train_x, y = data_train_y,
    batch_size = 1, epochs = 1, verbose = 0, shuffle = FALSE
  )
  lstm_model %>% reset_states()
}

stopCluster(cores.cluster)



# Prediction
x = predict(lstm_model, data_train_x, batch_size = 1)
y = predict(lstm_model, data_test_x, batch_size = 1)








