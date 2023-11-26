library(tidyverse)
library(tidymodels)
library(modeltime)

# 1. Read in data
train_lm <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm <- read_rds('data/avg_final_data/final_test_lm.rds')

complete_lm <- train_lm |> rbind(test_lm)

unique_countries <- unique(complete_lm$location)

# Empty list to store results for each country
country_results = list()

for (country in unique_countries) {
  # Filter data for the current country
  country_data = complete_lm %>% filter(location == country)
  
  # Split into train and test sets
  train_data = country_data %>% filter(date < as.Date("2023-01-01") & date >= as.Date("2020-01-04"))
  test_data = country_data %>% filter(date >= as.Date("2023-01-01"))
  
  # Define the rolling origin resampling
  rolling_origin_resampling = rolling_origin(
    train_data,
    initial = 365,  # Example: initial one year
    assess = 30,    # Example: assess one month at a time
    skip = 30,      # Example: skip one month before the next assessment
    cumulative = FALSE
  )
  
  # Define AutoARIMA Model
  autoarima_model = arima_reg() %>% set_engine('auto_arima')
  
  # Define Recipe
  autoarima_recipe = recipe(new_cases ~ date, data = train_data)
  
  # Define Workflow
  autoarima_wflow = workflow() %>%
    add_model(autoarima_model) %>%
    add_recipe(autoarima_recipe)
  
  # Fit the model across each fold
  autoarima_results = fit_resamples(
    autoarima_wflow,
    resamples = rolling_origin_resampling,
    control = control_resamples(verbose = TRUE) # Adjust control settings as needed
  )
  
  # Collect metrics
  metrics = autoarima_results %>% collect_metrics()
  
  # Make predictions on the test set
  final_model = finalize_workflow(autoarima_wflow, select_best(autoarima_results, "rmse"))
  fitted_model = fit(final_model, data = train_data)
  predictions = predict(fitted_model, new_data = test_data)
  
  # Store results
  country_results[[country]] <- list(
    Metrics = metrics,
    Predictions = predictions
  )
}

results_summary = map_df(names(country_results), ~{
  country_name = .x
  training_rmse = country_results[[country_name]]$Metrics %>% 
    filter(.metric == "rmse") %>% 
    pull(mean)
  
  # Assuming `value` is the actual cases in your test data
  test_data = complete_lm %>% filter(location == country_name, date >= as.Date("2023-01-01"))
  predictions = country_results[[country_name]]$Predictions
  testing_rmse = ModelMetrics::rmse(test_data$new_cases, predictions$.pred)
  
  tibble(location = country_name,
         train_rmse = round(training_rmse, 4),
         test_rmse = round(testing_rmse, 4))
})

country_results$`United States`$.pred |> 
  bind_rows(train_lm |> filter(location == "United States") |> select(date, new_cases))

ggplot() + 
  geom_line(aes(date, train_lm$new_cases)) + 
  geom_line(aes(date, country_results$`United States`$.pred)) 

