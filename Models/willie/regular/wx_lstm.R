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
final_train_nn = readRDS('data/avg_final_data/final_train_nn.rds')
final_test_nn = readRDS('data/avg_final_data/final_test_nn.rds')

resp_scale = c(mean(final_train_nn$new_cases), sd(final_train_nn$new_cases))

# Remove observations before first appearance of COVID: 2020-01-04
# choose predictors and normalize data
# lag 7 days
complete_nn = final_train_nn %>% rbind(final_test_nn) %>%
  filter(date >= as.Date("2020-01-04")) %>%
  group_by(location) %>%
  select(c(date, new_cases, location, new_deaths, new_tests, new_tests_b, gdp_per_capita)) %>%
  ungroup()

country_names = sort(unique(complete_nn$location))
country_convert = tibble(
  location = country_names,
  locid = 1:length(country_names)
)
complete_nn = complete_nn %>%
  left_join(country_convert, by = join_by(location == location))

data_norm = complete_nn %>%
  mutate(
    new_cases = as.vector(scale(new_cases)),
    new_deaths = as.vector(scale(new_deaths)),
    new_tests = as.vector(scale(new_tests)),
    gdp_per_capita = as.vector(scale(gdp_per_capita)),
    new_tests_b = ifelse(new_tests_b, 1, 0)
  ) %>%
  group_by(location) %>%
  mutate(
    # new_cases
    new_cases_lag7 = lag(new_cases, 7, default = NA),
    # location
    locid_lag7 = lag(locid, 7, default = NA),
    # new_deaths
    new_deaths_lag7 = lag(new_deaths, 7, default = NA),
    # new_tests
    new_tests_lag7 = lag(new_tests, 7, default = NA),
    # new_tests_b
    new_tests_b_lag7 = lag(new_tests_b, 7, default = NA),
    # gdp
    gdp_lag7 = lag(gdp_per_capita, 7, default = NA)
  ) %>%
  ungroup() %>%
  drop_na()

train_nn = data_norm %>% filter(date < as.Date("2023-01-01"))
test_nn = data_norm %>% filter(date >= as.Date("2023-01-01"))



# transform data into 3d array
# training predictors
data_train_x = array(0, dim = c(nrow(train_nn), 1, 6))
data_train_x[,,1] = as.matrix(train_nn %>% select(new_cases_lag7))
data_train_x[,,2] = as.matrix(train_nn %>% select(locid_lag7))
data_train_x[,,3] = as.matrix(train_nn %>% select(new_deaths_lag7))
data_train_x[,,4] = as.matrix(train_nn %>% select(new_tests_lag7))
data_train_x[,,5] = as.matrix(train_nn %>% select(new_tests_b_lag7))
data_train_x[,,6] = as.matrix(train_nn %>% select(gdp_lag7))
# training response
data_train_y = array(0, dim = c(nrow(train_nn), 1, 1))
data_train_y[,,1] = train_nn$new_cases





# testing predictors
data_test_x = array(0, dim = c(nrow(test_nn), 1, 6))
data_test_x[,,1] = as.matrix(test_nn %>% select(new_cases_lag7))
data_test_x[,,2] = as.matrix(test_nn %>% select(locid_lag7))
data_test_x[,,3] = as.matrix(test_nn %>% select(new_deaths_lag7))
data_test_x[,,4] = as.matrix(test_nn %>% select(new_tests_lag7))
data_test_x[,,5] = as.matrix(test_nn %>% select(new_tests_b_lag7))
data_test_x[,,6] = as.matrix(test_nn %>% select(gdp_lag7))
# testing response
data_test_y = array(0, dim = c(nrow(test_nn), 1, 1))
data_test_y[,,1] = test_nn$new_cases





# LSTM Model
# tuning parameters
num_units = 30
num_epochs = 10

# build
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)

lstm_model = keras_model_sequential()
lstm_model %>%
  layer_lstm(
    units = num_units, batch_input_shape = c(1, 1, 6), return_sequences = TRUE, stateful = TRUE
  ) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(
    units = num_units, return_sequences = TRUE, stateful = TRUE
  ) %>%
  layer_dense(units = 1)
# compile
lstm_model %>% compile(loss = "mse", optimizer = "adam", metrics = "mae")
summary(lstm_model)
for (i in 1:num_epochs) {
  lstm_model %>% fit(
    x = data_train_x, y = data_train_y,
    batch_size = 1, epochs = 1, verbose = 0, shuffle = FALSE
  )
  lstm_model %>% reset_states()
}

stopCluster(cores.cluster)



# Prediction and RMSE
library(ModelMetrics)
# Training set
fitted_init = predict(lstm_model, data_train_x, batch_size = 1) %>% .[,,1]
fitted_train = fitted_init * resp_scale[2] + resp_scale[1]
final_train = train_nn %>%
  mutate(
    pred = fitted_train,
    new_cases = new_cases * resp_scale[2] + resp_scale[1]
  )
results = final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)


# Testing set and
fitted_test_norm = predict(lstm_model, data_test_x, batch_size = 1) %>% .[,,1]
fitted_test = fitted_test_norm * resp_scale[2] + resp_scale[1]
final_test = test_nn %>%
  mutate(
    pred = fitted_test,
    new_cases = new_cases * resp_scale[2] + resp_scale[1]
  )
results_test = final_test %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)





# plots
x = final_train %>%
  filter(location == "Germany") %>%
  select(date, new_cases, pred) %>%
  pivot_longer(cols = c("new_cases", "pred"), names_to = "type", values_to = "value") %>%
  mutate(
    type = ifelse(type == 'new_cases', 'New Cases', 'Predicted New Cases'),
    type = factor(type, levels = c('New Cases', 'Predicted New Cases'))
  )



ggplot(x, aes(date, value)) +
  geom_line(aes(color = type, linetype = type)) +
  scale_y_continuous(n.breaks = 10) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  scale_color_manual(values = c("red", "blue")) +
  labs(
    title = "Training: Actual vs Predicted New Cases in Germany",
    x = "Date", y = "New Cases") +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 20),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#808080"))





y = final_test %>% 
  filter(location == "Germany") %>%
  select(date, new_cases, pred) %>%
  pivot_longer(cols = c("new_cases", "pred"), names_to = "type", values_to = "value") %>%
  mutate(
    type = ifelse(type == 'new_cases', 'New Cases', 'Predicted New Cases'),
    type = factor(type, levels = c('New Cases', 'Predicted New Cases'))
  )



ggplot(y, aes(date, value)) +
  geom_line(aes(color = type, linetype = type)) +
  scale_y_continuous(n.breaks = 10) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  scale_color_manual(values = c("red", "blue")) +
  labs(
    title = "Testing: Actual vs Predicted New Cases in Germany",
    x = "Date", y = "New Cases") +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 20),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#808080"))




