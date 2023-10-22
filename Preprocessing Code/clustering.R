# *****   Reference   *****
# https://www.tidyverse.org/blog/2022/12/tidyclust-0-1-0/
# https://uc-r.github.io/kmeans_clustering


library(cluster)
library(factoextra)
library(tidymodels)
library(tidyclust)

# 1. Determine optimal number of clusters

# training dataset / resamples

# Filter date For imputation for:
# total_tests           new_tests   positive_rate
# total_vaccinations    people_vaccinated
# For any other imputation, don't filter date

train_freena = na.omit(train_tree) %>%
  filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
  filter(!is.na(stringency_index))
folds = vfold_cv(train_freena, v = 5, repeats = 3)

# Define model
cluster_model = k_means(num_clusters = tune()) %>%
  set_engine("ClusterR")

# Define Recipe and workflow
cluster_recipe = recipe(~ stringency_index,
                        data = train_freena) %>%
  step_normalize(all_numeric_predictors())
cluster_wflow = workflow() %>%
  add_model(cluster_model) %>%
  add_recipe(cluster_recipe)

# Set up tuning grid
cluster_params = cluster_wflow %>%
  extract_parameter_set_dials() %>%
  update(num_clusters = num_clusters(c(2,8)))
cluster_grid = grid_regular(cluster_params, levels = 4)

# Tuning/Fitting
cluster_tuned = tune_cluster(
  cluster_wflow,
  resamples = folds,
  grid = cluster_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE),
  metrics = cluster_metric_set(silhouette_avg)
)

cluster_tuned %>% collect_metrics()
# Find the num_clusters where the mean is closest to 1





# 2. Predictions using clustering
cluster_model = k_means(num_clusters = 2) %>%
  set_engine("ClusterR")
cluster_wflow = workflow() %>%
  add_model(cluster_model) %>%
  add_recipe(cluster_recipe)

cluster_fit = fit(cluster_wflow, data = train_freena)

cluster_fit %>% extract_cluster_assignment()
# cluster_fit %>% extract_centroids()

cur_set = train_tree %>% filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01")))
final_set = cur_set %>%
  bind_cols(predict(cluster_fit, new_data = final_set))
# Rerun the code below, but replace mutate for each predictor
final_set = final_set %>%
  group_by(.pred_cluster) %>%
  mutate(total_vaccinations = ifelse(
    is.na(total_vaccinations),
    median(total_vaccinations, na.rm = TRUE),
    total_vaccinations)) %>%
  ungroup()
# This is the training set with missingness imputed between Feb 2021 to March 2022
# Merge this dataset with the training set outside the above date range to
# get the final training set



# View(final_set %>% skim_without_charts())
# View(train_tree %>%
#        filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
#        skim_without_charts())

# ggplot(final_set, aes(date, total_vaccinations)) +
#   geom_point(aes(color = location), alpha = 0.1)






