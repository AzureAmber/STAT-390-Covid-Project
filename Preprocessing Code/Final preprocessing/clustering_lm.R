# NOTE: Based on Willie's `clustering.R`

# *****   Reference   *****
# https://www.tidyverse.org/blog/2022/12/tidyclust-0-1-0/
# https://uc-r.github.io/kmeans_clustering

# Load packages  ----
library(tidyverse)
library(cluster)
library(factoextra)
library(tidymodels)
library(tidyclust)
library(doMC)
library(parallel)
library(doParallel)
theme_set(theme_minimal())
tidymodels_prefer()


# Load Data ----
train_lm <- read_rds("data/processed_data/train_lm.rds")
test_lm <- read_rds("data/processed_data/test_lm.rds")

# Looking at variables to use (want variance)
# For extreme_poverty --> not a good choice
ggplot(train_lm, aes(x = extreme_poverty)) + 
  geom_histogram(binwidth = 1) + 
  ggtitle("Distribution of Extreme Poverty")

# For population_density --> plausible?
ggplot(train_lm, aes(x = population_density)) + 
  geom_histogram(binwidth = 10) + 
  ggtitle("Distribution of Population Density")

# For life_expectancy --> not a good choice
ggplot(train_lm, aes(x = life_expectancy)) + 
  geom_histogram(binwidth = 10) + 
  ggtitle("Distribution of Life Expectancy")

# For diabetes prevalence --> not a good choice
ggplot(train_lm, aes(x = diabetes_prevalence)) + 
  geom_histogram(binwidth = 10) + 
  ggtitle("Distribution of Diabetes Prevalence")

# For female smokers --> plausible
ggplot(train_lm, aes(x = female_smokers)) + 
  geom_histogram(binwidth = 10) + 
  ggtitle("Distribution of Female Smokers")

# For male smokers --> plausible
ggplot(train_lm, aes(x = male_smokers)) + 
  geom_histogram(binwidth = 10) + 
  ggtitle("Distribution of Male Smokers")

# For stringency_index --> plausible
ggplot(train_lm, aes(x = stringency_index)) + 
  geom_histogram(binwidth = 10) + 
  ggtitle("Distribution of Stringency Index")

# Clustering Process ----
# 1. Determine optimal number of clusters

# detectCores(logical = FALSE)
# ***** INSERT YOUR NUMBER OF CORES HERE *****
# cores.cluster = makePSOCKcluster(10)
# registerDoParallel(cores.cluster)

# set up parallel processing foR mac
detectCores()
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# training dataset / resamples

# Filter date For imputation for:
# total_tests           new_tests   positive_rate
# total_vaccinations    people_vaccinated
# For any other imputation, don't filter date

train_freena = na.omit(train_lm) %>%
  filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
  filter(!is.na(life_expectancy) & !is.na(female_smokers) & !is.na(male_smokers))
folds = vfold_cv(train_freena, v = 5, repeats = 3)

# Define model
cluster_model = k_means(num_clusters = tune()) %>%
  set_engine("ClusterR")

# Define Recipe and workflow
# NOTE: trying other vars in addition to life_expectancy
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
  control = control_grid(parallel_over = "everything"),
  metrics = cluster_metric_set(silhouette_avg)
)

# stopCluster(cores.cluster)
stopCluster(cl)

cluster_tuned %>% collect_metrics()
# Find the num_clusters where the mean is closest to 1 --> 8 Clusters

# num_clusters .metric        .estimator  mean     n std_err .config             
#            4 silhouette_avg standard   0.479    15 0.00548 Preprocessor1_Model1
#            7 silhouette_avg standard   0.591    15 0.00991 Preprocessor1_Model2
#           10 silhouette_avg standard   0.719    15 0.00782 Preprocessor1_Model3
#           13 silhouette_avg standard   0.829    15 0.00446 Preprocessor1_Model4
#           16 silhouette_avg standard   0.934    15 0.00537 Preprocessor1_Model5

# 2. Predictions using clustering
train_freena = na.omit(train_lm) %>%
  filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
  filter(!is.na(life_expectancy) & !is.na(female_smokers) & !is.na(male_smokers))
cluster_recipe = recipe(~ life_expectancy + female_smokers + male_smokers,
                        data = train_freena) %>%
  step_normalize(all_numeric_predictors())
cluster_model = k_means(num_clusters = 10) %>% # using 10 clusters to stay consistent from before
  set_engine("ClusterR")
cluster_wflow = workflow() %>%
  add_model(cluster_model) %>%
  add_recipe(cluster_recipe)

cluster_fit = fit(cluster_wflow, data = train_freena)

cluster_fit %>% extract_cluster_assignment()
# cluster_fit %>% extract_centroids()

cur_set = train_lm
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
cur_set = test_lm
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

# write_rds(final_train %>% select(-c(.pred_cluster)), "data/finalized_data/final_train_lm.rds")
# write_rds(final_test %>% select(-c(.pred_cluster)), "data/finalized_data/final_test_lm.rds")






