library(cluster)
library(factoextra)
library(tidymodels)
library(tidyclust)
library(doParallel)

# detectCores(logical = FALSE)
# ***** INSERT YOUR NUMBER OF CORES HERE *****
cores.cluster = makePSOCKcluster(10)
registerDoParallel(cores.cluster)

# 1. Determine optimal number of clusters

# training dataset / resamples

# Filter date For imputation for:
# total_tests           new_tests   positive_rate   total_vaccinations
# For any other imputation, don't filter date

train_freena = na.omit(train_nn) %>%
  filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
  filter(!is.na(life_expectancy) & !is.na(female_smokers) & !is.na(male_smokers))
folds = vfold_cv(train_freena, v = 5, repeats = 3)

# Define model
cluster_model = k_means(num_clusters = tune()) %>%
  set_engine("ClusterR")

# Define Recipe and workflow
cluster_recipe = recipe(~ life_expectancy + female_smokers + male_smokers,
                        data = train_freena) %>%
  step_normalize(all_numeric_predictors())
cluster_wflow = workflow() %>%
  add_model(cluster_model) %>%
  add_recipe(cluster_recipe)

# Set up tuning grid
cluster_params = cluster_wflow %>%
  extract_parameter_set_dials() %>%
  update(num_clusters = num_clusters(c(4,16)))
cluster_grid = grid_regular(cluster_params, levels = 5)

# Tuning/Fitting
cluster_tuned = tune_cluster(
  cluster_wflow,
  resamples = folds,
  grid = cluster_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = cluster_metric_set(silhouette_avg)
)

stopCluster(cores.cluster)

cluster_tuned %>% collect_metrics()
# Find the num_clusters where the mean is closest to 1


# 2. Predictions using clustering
train_freena = na.omit(train_nn) %>%
  filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
  filter(!is.na(life_expectancy))

cluster_model = k_means(num_clusters = 10) %>%
  set_engine("ClusterR")
cluster_recipe = recipe(~ life_expectancy + female_smokers + male_smokers,
                        data = train_freena) %>%
  step_normalize(all_numeric_predictors())
cluster_wflow = workflow() %>%
  add_model(cluster_model) %>%
  add_recipe(cluster_recipe)

cluster_fit = fit(cluster_wflow, data = train_freena)

cluster_fit %>% extract_cluster_assignment()
# cluster_fit %>% extract_centroids()

cur_set = train_nn
final_train = cur_set %>%
  bind_cols(predict(cluster_fit, new_data = cur_set))
# View(final_train %>% skim_without_charts())

data_avg = final_train %>%
  group_by(.pred_cluster) %>%
  summarise(across(where(is.numeric), ~ median(.x, na.rm = TRUE)))

final_train %>%
  group_by(.pred_cluster) %>%
  summarise(v = paste(unique(location), collapse = ", "))

# Replace missingness for each numerical predictor by their cluster's median
library(rlang)
data_vars = colnames(cur_set)
for (i in data_vars) {
  if (class(cur_set[[i]]) == "numeric") {
    final_train <<- final_train %>%
      group_by(.pred_cluster) %>%
      mutate((!!sym(i)) := ifelse(
        is.na(!!sym(i)),
        median(!!sym(i), na.rm = TRUE),
        !!sym(i))) %>%
      ungroup()
  }
}
# View(final_train %>% skim_without_charts())

# Do the same for the testing set
cur_set = test_nn
final_test = cur_set %>%
  bind_cols(predict(cluster_fit, new_data = cur_set))
# View(final_test %>% skim_without_charts())

# Replace missingness for each numerical predictor by their cluster's median
for (i in data_vars) {
  if (class(cur_set[[i]]) == "numeric") {
    final_test <<- final_test %>%
      group_by(.pred_cluster) %>%
      mutate((!!sym(i)) := ifelse(
        is.na(!!sym(i)),
        median(!!sym(i), na.rm = TRUE),
        !!sym(i))) %>%
      ungroup()
  }
}
# View(final_test %>% skim_without_charts())

# Replace the remaining missingness with median values by clustering from training set.
for (i in seq(1, nrow(final_test))) {
  if (is.na(final_test$stringency_index[i])) {
    final_test$stringency_index[i] =
      (data_avg %>% filter(.pred_cluster == final_test$.pred_cluster[i]))$stringency_index
  }
}
# View(final_test %>% skim_without_charts())

# write_rds(final_train %>% select(-c(.pred_cluster)), "data/finalized_data/final_train_nn.rds")
# write_rds(final_test %>% select(-c(.pred_cluster)), "data/finalized_data/final_test_nn.rds")




