# *****   Reference   *****
# https://www.tidyverse.org/blog/2022/12/tidyclust-0-1-0/
# https://uc-r.github.io/kmeans_clustering


# IMPORTANT NOTES
# On line 18, replace 10 with your amount of cores
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
# total_tests           new_tests   positive_rate
# total_vaccinations    people_vaccinated
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
cluster_model = k_means(num_clusters = 16) %>%
  set_engine("ClusterR")
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
temp_set = train_tree %>% filter(date < as.Date("2021-02-01") | date > as.Date("2022-03-01"))
final_train = temp_set %>% bind_rows(final_set %>% select(-c(.pred_cluster)))
# Do the same for the testing set

# View(train_tree %>%
#        filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
#        skim_without_charts())

# ggplot(final_set %>% filter(total_tests <= 5e14), aes(date, total_tests)) +
#   geom_point(aes(color = location), alpha = 0.1)






