library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(forecast)
library(lubridate)
library(RcppRoll)

# 1. Read in data
train_lm <- read_rds('./data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('./data/finalized_data/final_test_lm.rds')


# weekly rolling average
train_lm <- train_lm %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), new_cases, value)) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(which(row_number() %% 7 == 1)) %>%
  mutate(seasonality_group = row_number() %% 53) %>%
  ungroup() %>%
  mutate(seasonality_group = as.factor(seasonality_group))
test_lm <- test_lm %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(value = roll_mean(new_cases, 7, align = "right", fill = NA)) %>%
  mutate(value = ifelse(is.na(value), new_cases, value)) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(which(row_number() %% 7 == 1)) %>%
  mutate(seasonality_group = row_number() %% 53) %>%
  ungroup() %>%
  mutate(seasonality_group = as.factor(seasonality_group))
complete_lm <- train_lm %>% rbind(test_lm) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()

ggplot(train_lm %>% filter(location == "United States"), aes(date, value)) +
  geom_point()

# 2. Find model trend by country
train_lm_fix <- NULL
test_lm_fix <- NULL
country_names <- unique(train_lm$location)
for (i in country_names) {
  data = train_lm %>% filter(location == i)
  complete_data = complete_lm %>% filter(location == i)
  # find linear model by country
  lm_model = lm(value ~ 0 + time(date) + seasonality_group, data)
  x = complete_data %>%
    mutate(
      trend = predict(lm_model, newdata = complete_data),
      slope = as.numeric(coef(lm_model)["time(date)"]),
      seasonality_add = trend - slope * time(date),
      err = value - trend) %>%
    mutate_if(is.numeric, round, 5)
  train_lm_fix <<- rbind(train_lm_fix, x %>% filter(date < as.Date("2023-01-01")))
  test_lm_fix <<- rbind(test_lm_fix, x %>% filter(date >= as.Date("2023-01-01")))
}
# plot of original data and trend
ggplot(train_lm_fix %>% filter(location == "United States")) +
  geom_line(aes(date, value), color = 'blue') +
  geom_line(aes(date, trend), color = 'red')+
  theme_minimal()
# plot of residual errors
ggplot(x %>% filter(location == "United States"), aes(date, err)) + geom_line()




# ARIMA Model tuning for errors
# 3. Create validation sets for every year train + 2 month test with 4-month increments
train_lm_fix_us <- train_lm_fix %>% filter(location == "United States")


# 4. Define model, recipe, and workflow
arima_model <- arima_reg(
  seasonal_period = 53,
  non_seasonal_ar = 2, non_seasonal_differences = 0, non_seasonal_ma = 0) %>%
  set_engine('arima')

arima_recipe <- recipe(err ~ date, data = train_lm_fix_us)
#arima_recipe %>% prep() %>% bake(new_data = NULL)


arima_wflow <- workflow() %>%
  add_model(arima_model) %>%
  add_recipe(arima_recipe)


arima_fit_us <- fit(arima_wflow, data = train_lm_fix_us)

final_train_us <- train_lm_fix_us %>%
  bind_cols(pred_err = arima_fit_us$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)


# error model
final_train_us %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = err, color = "train_actual")) + 
  geom_line(aes(y = pred_err, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "Error Model (US)",
       y = "Value", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)
# prediction model
final_train_us %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (US)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

library(ModelMetrics)
ModelMetrics::rmse(final_train_us$err, final_train_us$pred_err) #29534.38
ModelMetrics::rmse(final_train_us$value, final_train_us$pred)



## Germany

train_lm_fix_germ <- train_lm_fix %>% filter(location == "Germany")
test_lm_fix_germ <- test_lm_fix %>% filter(location == "Germany")

arima_fit_germ <- fit(arima_wflow, data = train_lm_fix_germ)

arima_fit_germ_test <- fit(arima_wflow, data = test_lm_fix_germ)


final_train_germ <- train_lm_fix_germ %>%
  bind_cols(pred_err = arima_fit_germ$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

final_test_germ <- test_lm_fix_germ %>%
  bind_cols(pred_err = arima_fit_germ_test$fit$fit$fit$data$.fitted) %>%
  mutate(pred = trend + pred_err) %>%
  mutate_if(is.numeric, round, 5)

final_test_germ %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, color = "train_actual")) + 
  geom_line(aes(y = pred, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (Germany)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)


ModelMetrics::rmse(final_test_germ$err, final_test_germ$pred_err) #6628.222
ModelMetrics::rmse(final_test_germ$value, final_test_germ$pred)

ModelMetrics::rmse(final_train_germ$err, final_train_germ$pred_err) #6628.222
ModelMetrics::rmse(final_train_germ$value, final_train_germ$pred)
