library(tidyverse)
library(tidymodels)
library(doParallel)
library(parallel)
library(tictoc)
library(ranger)


# Setup parallel processing
# detectCores() # 8
cores.cluster <- makePSOCKcluster(5)
registerDoParallel(cores.cluster)


# 1. Read in data
train_tree <- read_rds('data/finalized_data/final_train_tree.rds')
test_tree <- read_rds('data/finalized_data/final_test_tree.rds')

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree,
  initial = 366,
  assess = 30*2,
  skip = 30*4,
  cumulative = FALSE
)


# 3. Define model, recipe, and workflow
rf_model <- rand_forest(
  # start with 500 trees instead of tuning (match w bt)
  trees = 500,
  min_n = tune(), 
  mtry = tune()) |> 
  set_engine('ranger', importance = "impurity") %>%
  set_mode('regression')

rf_recipe <- recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())

# View(rf_recipe %>% prep() %>% bake(new_data = NULL))

rf_wflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rf_recipe)

# 4. Setup tuning grid
rf_params <- rf_wflow %>%
  extract_parameter_set_dials() %>%
  # mtry is up to # of predictors (30)
  update(mtry = mtry(c(5, 30)))

rf_grid <- grid_regular(rf_params, levels = 3)

# 5. Model Tuning
tic.clearlog()
tic('rf')

rf_tuned = tune_grid(
  rf_wflow,
  resamples = data_folds,
  grid = rf_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)


toc(log = TRUE)
time_log <- tic.log(format = FALSE)
rf_tictoc <- tibble(model = time_log[[1]]$msg, 
                       runtime = time_log[[1]]$toc - time_log[[1]]$tic)
stopCluster(cores.cluster)


save(rf_tuned, rf_tictoc, file = "Models/cindy/results/rf_tuned_1.rda")

# 6. Review the best results (lower RMSE is better)
load("Models/cindy/results/rf_tuned_1.rda")
show_best(rf_tuned, metric = "rmse")
# mtry min_n .metric .estimator   mean     n std_err .config             
#   30     2 rmse    standard    9638.   205   1147. Preprocessor1_Model3
#   17     2 rmse    standard   10690.   205   1215. Preprocessor1_Model2
#   30    21 rmse    standard   13689.   205   1590. Preprocessor1_Model6
#   17    21 rmse    standard   14852.   205   1616. Preprocessor1_Model5
#   30    40 rmse    standard   15231.   205   1683. Preprocessor1_Model9

# NOTE: big mtry, small min_n, increase trees 2nd round!
autoplot(rf_tuned, metric = "rmse")

#7. Fit best model
# trees = 1000, mtry = 30, min_n = 2
rf_model_new <- rand_forest(
# increasing number of trees from 500 --> 1000
  trees = 1000,
  min_n = 2, 
  mtry = 30) |> 
  set_engine('ranger', importance = "impurity") %>%
  set_mode('regression')

rf_wflow_new <- workflow() %>%
  add_model(rf_model_new) %>%
  add_recipe(rf_recipe)

rf_fit <- fit(rf_wflow_new, data = train_tree)

rf_train_results <- train_tree %>%
  bind_cols(predict(rf_fit, new_data = train_tree)) %>%
  rename(pred = .pred)

rf_train_tibble <- rf_train_results |> 
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location) |> 
  print(n = 23) |> 
  pivot_wider(names_from = location, values_from = value)

view(rf_train_tibble)

# location         value
# <chr>            <dbl>
# 1 Argentina       1939. 
# 2 Australia       3467. 
# 3 Canada          2809. 
# 4 Colombia         416. 
# 5 Ecuador         4191. 
# 6 Ethiopia          16.7
# 7 France         11177. 
# 8 Germany         6261. 
# 9 India           3593. 
# 10 Italy            589. 
# 11 Japan           1785. 
# 12 Mexico          2008. 
# 13 Morocco           30.7
# 14 Pakistan          49.2
# 15 Philippines      134. 
# 16 Russia           338. 
# 17 Saudi Arabia      11.1
# 18 South Africa     110. 
# 19 South Korea     2647. 
# 20 Sri Lanka        469. 
# 21 Turkey          1381. 
# 22 United Kingdom   833. 
# 23 United States   6882. 

rf_pred <- predict(rf_fit, new_data = test_tree) %>% 
  bind_cols(test_tree %>% select(new_cases, location, date)) |> 
  rename(pred = .pred) |> 
  

rf_test_tibble <- rf_pred |> 
  group_by(location) |> 
  summarise(value = ModelMetrics::rmse(new_cases, pred)) |>
  arrange(location) |> 
  print(n = 23) |> 
  pivot_wider(names_from = location, values_from = value)

view(rf_test_tibble)

# 8. Looking at results ----
# Actual vs. Pred per country
# Define the lists of countries
g20 <- c('Argentina', 'Australia', 'Canada', 'France', 'Germany',
         'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
         'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')

g24 <- c('Argentina', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
         'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')

# Combine and get unique countries from both lists
unique_countries <- unique(c(g20, g24))


## Training Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Training: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/random_forest/training_plots/rf_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  rf_country <- ggplot(rf_train_results %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "rand_forest(trees = 1000, mtry = 30, min_n = 2)",
         caption = "Ranger",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, rf_country, width = 10, height = 6)
}

## Testing Actual vs. Pred plots ----
# Loop through each country
for(country in unique_countries) {
  # Create plot for the current country
  plot_name <- paste("Testing: Actual vs. Predicted New Cases in", country, "in 2023")
  file_name <- paste("Results/cindy/random_forest/testing_plots/rf_", gsub(" ", "_", tolower(country)), ".jpeg", sep = "")
  
  rf_country <- ggplot(rf_pred %>% filter(location == country)) +
    geom_line(aes(x = date, y = new_cases, color = "Actual New Cases")) +
    geom_line(aes(x = date, y = pred, color = "Predicted New Cases"), linetype = "dashed") +
    scale_y_continuous(n.breaks = 15) + 
    scale_x_date(date_breaks = "2 months", date_labels = "%b %y") +
    theme_minimal() + 
    labs(x = "Date", 
         y = "New Cases", 
         title = plot_name,
         subtitle = "rand_forest(trees = 1000, mtry = 30, min_n = 2)",
         caption = "Ranger",
         color = "") + 
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(face = "italic", hjust = 0.5),
          legend.position = "bottom",
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))
  
  
  # Save the plot with specific dimensions
  ggsave(file_name, rf_country, width = 10, height = 6)
}

# Table for test predictions
rf_results <- rf_pred %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)

rf_results |> print(n = 23)

# location          value
# <chr>             <dbl>
# 1 Argentina       90715. 
# 2 Australia       43025. 
# 3 Canada           1352. 
# 4 Colombia         1281. 
# 5 Ecuador          4606. 
# 6 Ethiopia          361. 
# 7 France          20251. 
# 8 Germany         17946. 
# 9 India           92999. 
# 10 Italy          124514. 
# 11 Japan           22248. 
# 12 Mexico           6961. 
# 13 Morocco         10151. 
# 14 Pakistan           95.6
# 15 Philippines      9541. 
# 16 Russia           9466. 
# 17 Saudi Arabia       48.7
# 18 South Africa       71.7
# 19 South Korea     24795. 
# 20 Sri Lanka          25.4
# 21 Turkey           5224. 
# 22 United Kingdom  40421. 
# 23 United States  558589.

save(rf_train_results, rf_pred, rf_results, file = "Models/cindy/results/rf.rda")
