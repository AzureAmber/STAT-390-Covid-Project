library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(forecast)
library(lubridate)


# Source
# https://www.r-bloggers.com/2020/06/introducing-modeltime-tidy-time-series-forecasting-using-tidymodels/
# https://www.youtube.com/watch?v=8na-sasmu5I



# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)


# 1. Read in data
train_lm <- read_rds('./data/finalized_data/final_train_lm.rds')
test_lm <- read_rds('./data/finalized_data/final_test_lm.rds')

# 2. filter out the two countries we will be use

## stationary country rep
train_us <- train_lm %>% 
  filter(location == "United States")
test_us <- test_lm %>% 
  filter(location == "United States")


# 3. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_us,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)
#data_folds


# # 4. automatic grid search for best p,d,q
# 
# # us: 3,1,5
# train_us %>% 
#   select(new_cases) %>% 
#   ts() %>% 
#   auto.arima() %>% 
#   summary()
# 
# # germany:2,1,2
# train_germ %>% 
#   select(new_cases) %>% 
#   ts() %>% 
#   auto.arima() %>% 
#   summary()

# # 5. manual grid search for best p,d,q
# 
# p_range <- 0:5
# d_range <- 0:2
# q_range <- 0:5
# 
# best_aic <- Inf
# best_order <- c(0,0,0)
# best_model <- NULL
# 
# for (p in p_range){
#   for(d in d_range){
#     for(q in q_range){
#       model <- tryCatch(Arima(train_germ, order = c(p,d,q)),
#                         error = function(e) NULL)
#       
#       if(!is.null(model)){
#         if (model$aic < best_aic){
#           best_aic <- model$aic
#           best_order <- c(p, d, q)
#           best_model <- model
#         }
#       }
#     }
#   }
# }
# 
# 
# best_order
# best_aic
# 
# forecast <- forecast(best_model, h = 12)
# 
# plot(forecast)
# 
# # 6. compare manual search and automatic search results by AIC
# 
# manual_model_us <- Arima(train_us$new_cases, order = c(0,0,0))
# print(AIC(manual_model_us)) #28834.8
# 
# auto_model_us <- auto.arima(train_us$new_cases)
# print(AIC(auto_model)) #26758.13
# 
# 
# manual_model_germ <- Arima(train_germ$new_cases, order = c(0,0,0))
# print(AIC(manual_model_germ)) #29248.58
# 
# auto_model_germ <- auto.arima(train_germ$new_cases)
# print(AIC(auto_model_germ)) #28719.69

# 7. train the model using automatic search result (smaller AIC value)

arima_us <- arima_reg(
  seasonal_period = "auto",
  non_seasonal_ar = 3, non_seasonal_differences = 1, non_seasonal_ma = 5,
  seasonal_ar = tune(), seasonal_differences = tune(), seasonal_ma = tune()) %>%
  set_engine('arima')

arima_recipe <- recipe(new_cases ~ date, data = train_us)

# View(arima_recipe %>% prep() %>% bake(new_data = NULL))

arima_wflow <- workflow() %>%
  add_model(arima_us) %>%
  add_recipe(arima_recipe)

save(arima_wflow, file = "Models/erica/results/arima_us_wflow.rda" )

# 4. Setup tuning grid
arima_params <- arima_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    seasonal_ar = seasonal_ar(c(0, 2)),
    seasonal_ma = seasonal_ma(c(0, 2)),
    seasonal_differences = seasonal_differences(c(0,1))
  )


arima_grid = grid_regular(arima_params, levels = 3)

# 5. Model Tuning
arima_tuned_us <- tune_grid(
  arima_wflow,
  resamples = data_folds,
  grid = arima_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)

arima_tuned_us %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)

save(arima_tuned_us, file = "Models/erica/results/arima_tuned_us_1.rda")

# 6. Results
arima_us_autoplot <- autoplot(arima_tuned_us, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/arima_us_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
# Print the plot to the device
print(arima_us_autoplot)
# Close the device
dev.off()


show_best(arima_tuned_us, metric = "rmse") #best with RMSE: 63021

## BEST MODEL HYPERPARAMTER: p,d,q: 3,1,5; P,D,Q: 2,0,1


# 7. fit train and predict test

arima_wflow_tuned <- arima_wflow %>%
  finalize_workflow(select_best(arima_tuned_us, metric = "rmse"))

arima_fit <- fit(arima_wflow_tuned, train_us)

#training set prediction & graph
train_predictions <- predict(arima_fit, new_data = train_us) %>%
  bind_cols(train_us %>% select(date, new_cases)) %>%
  mutate(estimate = .pred) %>%
  select(date, new_cases, estimate)

train_predictions %>% 
  yardstick::rmse(new_cases, estimate) #RMSE: 133255

train_prediction_us_plot <- train_predictions %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = new_cases, color = "train_actual")) + 
  geom_line(aes(y = estimate, color = "train_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("train_actual" = "red", "train_pred" = "blue"),
                     name = "Data", 
                     labels = c("train_actual" = "Train Actual", "train_pred" = "Train Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (US)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

ggsave("Models/erica/results/train_prediction_us_plot.jpeg", train_prediction_us_plot, width = 8, height = 6, dpi = 300)



#test set prediction & graph

test_predictions <- predict(arima_fit, new_data = test_us) %>% 
  bind_cols(test_us %>% select(date, new_cases))%>% 
  mutate(estimate = .pred) %>% 
  select(date, new_cases, estimate)

test_predictions %>% 
  yardstick::rmse(new_cases, estimate) #RMSE: 71678

test_prediction_us_plot <- test_predictions %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = new_cases, color = "test_actual")) + 
  geom_line(aes(y = estimate, color = "test_pred"), linetype = "dashed") + 
  scale_color_manual(values = c("test_actual" = "red", "test_pred" = "blue"),
                     name = "Data", 
                     labels = c("test_actual" = "Test Actual", "test_pred" = "Test Predicted")) +
  labs(title = "ARIMA Model Fit vs Actual Data (US)",
       y = "New Cases", x = "Date") +
  theme_minimal() +
  scale_y_continuous(n.breaks = 15)

ggsave("Models/erica/results/test_prediction_us_plot.jpeg", test_prediction_us_plot, width = 8, height = 6, dpi = 300)

## for all the other countries that data are stationary

# first extract countries 
locations <- unique(train_lm$location)

for (loc in locations) {
  location_data <- train_lm %>% filter(location == loc)
  location_name <- paste0("train_", make.names(loc))  # Create the name with "train_" prefix
  assign(location_name, location_data, envir = .GlobalEnv)
}

for (loc in locations) {
  location_data <- test_lm %>% filter(location == loc)
  location_name <- paste0("test_", make.names(loc))  # Create the name with "train_" prefix
  assign(location_name, location_data, envir = .GlobalEnv)
}

###### use the same hyperparamter to fit other countries' data

#non-stationary countries, so we remove those
countries_of_interest <- c("Australia", "France", "Germany",
                           "Japan", "Sri Lanka", "Turkey")

locations <- setdiff(unique(train_lm$location), countries_of_interest)

fitted_models <- list()

# Loop through each location, fit the model, and store the result
for (loc in locations) {
  
  train_df_name <- paste0("train_", make.names(loc))
  
  if (exists(train_df_name)) {
    
    train_data <- get(train_df_name)
    
    fitted_model <- fit(arima_wflow_tuned, data = train_data)
    
    fitted_models[[loc]] <- fitted_model
  }
}


# Prepare a list to store the RMSE values for each location
rmse_values <- list()

# Loop through each fitted model and calculate predictions on the training set
for (loc in names(fitted_models)) {

  fitted_model <- fitted_models[[loc]]
  
  train_df_name <- paste0("train_", make.names(loc))
  
  if (exists(train_df_name)) {
    
    train_data <- get(train_df_name)
    
    train_predictions <- predict(fitted_model, new_data = train_data) %>%
      bind_cols(train_data %>% select(date, new_cases)) %>%
      mutate(estimate = .pred) %>%
      select(date, new_cases, estimate)
    
    rmse_value <- train_predictions %>%
      yardstick::rmse(truth = new_cases, estimate = estimate)
    
    rmse_values[[loc]] <- rmse_value
  }
}

rmse_tibble <- tibble(
  location = names(rmse_values),
  rmse = sapply(rmse_values, function(rmse_df) { rmse_df$.estimate })
)

print(rmse_tibble) %>% 
  arrange(rmse)


# location          rmse
# <chr>            <dbl>
#   1 Ethiopia          742.
# 2 Saudi Arabia     1290.
# 3 Ecuador          1582.
# 4 Pakistan         2198.
# 5 Morocco          2245.
# 6 South Africa     6236.
# 7 Canada           6644.
# 8 Philippines      6699.
# 9 Colombia         8943.
# 10 Mexico          10093.
# 11 Argentina       20050.
# 12 Russia          33104.
# 13 Italy           34479.
# 14 United Kingdom  35621.
# 15 South Korea     74888.
# 16 India           83228.
# 17 United States  133255.













