library(tidyverse)
library(tidymodels)
library(doParallel)

# Source
# https://juliasilge.com/blog/xgboost-tune-volleyball/


# 1. Read in data
final_train_tree = readRDS('data/avg_final_data/final_train_tree.rds')
final_test_tree = readRDS('data/avg_final_data/final_test_tree.rds')

# add lagged predictors
# Remove observations before first appearance of COVID: 2020-01-04
complete_tree = final_train_tree %>% rbind(final_test_tree) %>%
  filter(date >= as.Date("2020-01-04")) %>%
  group_by(location) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(
    one_lag_wk = lag(new_cases, n = 7, default = 0),
    two_lag_wk = lag(new_cases, n = 14, default = 0),
    one_lag_month = lag(new_cases, n = 30, default = 0)
  )
train_tree = complete_tree %>% filter(date < as.Date("2023-01-01")) %>%
  group_by(date) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()
test_tree = complete_tree %>% filter(date >= as.Date("2023-01-01")) %>%
  group_by(date) %>%
  arrange(date, .by_group = TRUE) %>%
  ungroup()

# 2. Create validation sets for every year train + 2 month test with 4-month increments
data_folds = rolling_origin(
  train_tree,
  initial = 23*366,
  assess = 23*30*2,
  skip = 23*30*4,
  cumulative = FALSE
)
data_folds





# 3. Define model, recipe, and workflow
btree_model = boost_tree(
    trees = tune(), tree_depth = tune(),
    learn_rate = tune(), min_n = tune(), mtry = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

btree_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
# View(btree_recipe %>% prep() %>% bake(new_data = NULL))

btree_wflow = workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

# 4. Setup tuning grid
btree_params = btree_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    trees = trees(c(500, 1000)),
    min_n = min_n(c(5,15)),
    mtry = mtry(c(5,25)),
    tree_depth = tree_depth(c(2,20))
  )
btree_grid = grid_regular(btree_params, levels = 3)

# 5. Model Tuning
# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(20)
registerDoParallel(cores.cluster)

btree_tuned = tune_grid(
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

# 6. Results
autoplot(btree_tuned, metric = "rmse")





# 7. Fit Best Model
btree_model = boost_tree(
  trees = 1000, tree_depth = 2,
  learn_rate = 0.316, min_n = 10, mtry = 25) %>%
  set_engine('xgboost') %>%
  set_mode('regression')
btree_recipe = recipe(new_cases ~ ., data = train_tree) %>%
  step_rm(date) %>%
  step_mutate(
    G20 = ifelse(G20, 1, 0),
    G24 = ifelse(G24, 1, 0)) %>%
  step_dummy(all_nominal_predictors())
btree_wflow = workflow() %>%
  add_model(btree_model) %>%
  add_recipe(btree_recipe)

btree_fit = fit(btree_wflow, data = train_tree)
final_train = train_tree %>%
  bind_cols(predict(btree_fit, new_data = train_tree)) %>%
  rename(pred = .pred)
final_test = test_tree %>%
  bind_cols(predict(btree_fit, new_data = test_tree)) %>%
  rename(pred = .pred)


library(ModelMetrics)
results = final_train %>%
  group_by(location) %>%
  summarise(value = rmse(new_cases, pred)) %>%
  arrange(location)
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





