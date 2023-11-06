# *****   Reference   *****
# https://www.tidyverse.org/blog/2022/12/tidyclust-0-1-0/
# https://uc-r.github.io/kmeans_clustering


# IMPORTANT NOTES
# On line 19, replace 10 with your amount of cores
# On lines 31, 86, and 109, replace train_tree with your training set


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

train_freena = na.omit(train_tree) %>%
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
train_freena = na.omit(train_tree) %>%
  filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
  filter(!is.na(life_expectancy) & !is.na(female_smokers) & !is.na(male_smokers))

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

cur_set = train_tree %>% filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01")))
final_set = cur_set %>%
  bind_cols(predict(cluster_fit, new_data = cur_set))
# View(final_set %>% skim_without_charts())

data_avg = final_set %>%
  group_by(.pred_cluster) %>%
  summarise(across(where(is.numeric), ~ median(.x, na.rm = TRUE)))

final_set %>%
  group_by(.pred_cluster) %>%
  summarise(v = paste(unique(location), collapse = ", "))

# Replace missingness for each numerical predictor by their cluster's median
library(rlang)
data_vars = colnames(cur_set)
for (i in data_vars) {
  if (class(cur_set[[i]]) == "numeric") {
    final_set <<- final_set %>%
      group_by(.pred_cluster) %>%
      mutate((!!sym(i)) := ifelse(
        is.na(!!sym(i)),
        median(!!sym(i), na.rm = TRUE),
        !!sym(i))) %>%
      ungroup()
  }
}

# This is the training set with missingness imputed between Feb 2021 to March 2022
# Merge this dataset with the training set outside the above date range to
# get the final training set
temp_set = train_tree %>%
  filter(date < as.Date("2021-02-01") | date > as.Date("2022-03-01"))
temp_set = temp_set %>%
  bind_cols(predict(cluster_fit, new_data = temp_set))
# View(temp_set %>% skim_without_charts())
for (i in seq(1, nrow(temp_set))) {
  if (is.na(temp_set$new_cases[i])) {
    temp_set$new_cases[i] =
      (data_avg %>% filter(.pred_cluster == temp_set$.pred_cluster[i]))$new_cases
  }
}
v = 1e15
temp_set = replace(temp_set, is.na(temp_set), v)
final_train = temp_set %>% bind_rows(final_set) %>% select(-c(.pred_cluster))
# View(final_train %>% skim_without_charts())



# Do the same for the testing set
final_test = test_tree %>%
  bind_cols(predict(cluster_fit, new_data = test_tree))
# View(final_test %>% skim_without_charts())
for (i in seq(1, nrow(final_test))) {
  if (is.na(final_test$new_cases[i])) {
    final_test$new_cases[i] =
      (data_avg %>% filter(.pred_cluster == final_test$.pred_cluster[i]))$new_cases
  }
}
final_test = replace(final_test, is.na(final_test), v)
# View(final_test %>% skim_without_charts())



# write_rds(final_train, "data/finalized_data/final_train_tree.rds")
# write_rds(final_test, "data/finalized_data/final_test_tree.rds")



