library(tidyverse)
library(tidymodels)
library(doParallel)
library(dplyr)
library(ModelMetrics)



tidymodels_prefer()

# Source
# https://juliasilge.com



# 1. Read in data
train_tree <- readRDS('Data/avg_final_data/final_train_tree.rds')
test_tree <- readRDS('Data/avg_final_data/final_test_tree.rds')

## add lagged predictor variables & remove observations before first COVID:
complete_tree <- train_tree %>% rbind(test_tree) %>% 
  filter(date >= as.Date("2020-01-19")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  mutate(
    new_cases_log = ifelse(is.finite(log(new_cases)), log(new_cases), 0),
    one_wk_lag = dplyr::lag(new_cases_log, n = 7, default = 0),
    two_wk_lag = dplyr::lag(new_cases_log, n = 14, default = 0),
    one_month_lag = dplyr::lag(new_cases_log, n = 30, default =0)
  )


train_tree <- complete_tree %>% 
  filter(date < as.Date("2023-01-01")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()

test_tree <- complete_tree %>% 
  filter(date >= as.Date ("2023-01-01")) %>% 
  group_by(location) %>% 
  arrange(date, .by_group = TRUE) %>% 
  ungroup()

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds <- rolling_origin(
  train_tree,
  initial = 366*23,
  assess = 30*2*23,
  skip = 30*4*23,
  cumulative = FALSE
)
#data_folds

# 3. Define model, recipe, and workflow
btree_recipe <- recipe(new_cases_log ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())


btree_model <- boost_tree(
  trees = 1000, tree_depth = tune(),
  learn_rate = tune(), min_n = tune(), mtry = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_wflow <- workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid


btree_params <- btree_wflow %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1,5)),
         tree_depth = tree_depth(c(20,45)),
         learn_rate = learn_rate(c(-1,0)),
         min_n = min_n(c(1,3))
  )

btree_grid <- grid_regular(btree_params, levels = 3)

# 5. Model Tuning

# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)

btree_tuned <- tune_grid(
  btree_wflow,
  resamples = data_folds,
  grid = btree_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything"),
  metrics = metric_set(yardstick::rmse)
)

stopCluster(cores.cluster)

btree_tuned %>% collect_metrics() %>%
  group_by(.metric) %>%
  arrange(mean)


# 6. results
btree_autoplot <- autoplot(btree_tuned, metric = "rmse")
show_best(btree_tuned, metric = "rmse")


jpeg("Models/erica/results/xgboost/xgboost_log_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
print(btree_autoplot)
dev.off()


# 7. Fit Best Model

# mtry = 5, min_n = 3, tree_depth = 20, learn_rate = 0.1, trees = 1000

btree_model <- boost_tree(
  trees = 1000, 
  tree_depth = 20,
  learn_rate = 0.1, 
  min_n = 3, 
  mtry = 5) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_recipe <- recipe(new_cases_log ~ ., data = train_tree) %>%
   step_rm(date) %>%
   step_mutate(
     G20 = ifelse(G20, 1, 0),
     G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

btree_wflow_tuned <- workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)
 
btree_fit <- fit(btree_wflow_tuned, data = train_tree)


#predictions on country level
final_btree_train <- train_tree %>%
  bind_cols(predict(btree_fit, new_data = train_tree)) %>%
  mutate(.pred = exp(.pred)) %>% 
  rename(pred = .pred)


btree_log_train_results <- final_btree_train %>%
  group_by(location) %>%
  summarize(rmse_train_pred = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location)


final_btree_test <- test_tree %>% 
  bind_cols(predict(btree_fit, new_data = test_tree)) %>% 
  mutate(.pred = exp(.pred)) %>% 
  rename(pred = .pred)

btree_log_test_results <- final_btree_test %>% 
  group_by(location) %>% 
  summarize(rmse_test_pred = ModelMetrics::rmse(new_cases, pred)) %>% 
  arrange(location)

btree_log_results <- btree_log_train_results %>% 
   inner_join(btree_log_test_results, by = "location", suffix = c("rmse_train_pred", "rmse_test_pred"))

write.csv(btree_log_results, "Results/erica/xgboost_log/xgboost_log_rmse_results.csv", row.names = FALSE)
 



## Training Visualization

train_plot <- final_btree_train %>%
  filter(location == "South Korea") %>%
  ggplot(aes(x=date))+
  geom_line(aes(y = new_cases, color = "Actual New Cases"))+
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")+
  scale_y_continuous(n.breaks = 15)+
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y")+
  theme_minimal()+
  labs(x = "Date",
       y = "New Cases",
       title = "Training: Actual vs. Predicted New Cases in South Korea in 2023",
       subtitle = "boost_tree(trees = 1000, tree_depth = 20, learn_rate = 0.1, min_n = 3, mtry = 5",
       caption = "xgboost",
       color = "") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(train_plot, file = "Results/erica/xgboost_log/South Korea_train_pred.jpeg", width=8, height =7, dpi = 300)



## Testing visualization

test_plot <- final_btree_test %>%
  filter(location == "United States") %>%
  ggplot(aes(x=date)) +
  geom_line(aes(y = new_cases, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() +
  labs(x = "Date",
       y = "New Cases",
       title = "Testing: Actual vs. Predicted New Cases in United States in 2023",
       subtitle = "boost_tree(trees = 1000, tree_depth = 20, learn_rate = 0.1, min_n = 3, mtry = 5)",
       caption = "xgboost",
       color = "") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(test_plot, file = "Results/erica/xgboost_log/United States_test_pred.jpeg", width=8, height =7, dpi = 300)




