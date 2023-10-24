# NOTE: Based on Willie's `clustering.R`

# *****   Reference   *****
# https://www.tidyverse.org/blog/2022/12/tidyclust-0-1-0/
# https://uc-r.github.io/kmeans_clustering

# Load packages  ----
library(cluster)
library(factoextra)
library(tidymodels)
library(tidyclust)
theme_set(theme_minimal())

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

# training dataset / resamples

# Filter date For imputation for:
# total_tests           new_tests   positive_rate
# total_vaccinations    people_vaccinated
# For any other imputation, don't filter date

train_freena = na.omit(train_lm) %>%
  filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
  filter(!is.na(life_expectancy))
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
# Find the num_clusters where the mean is closest to 1 --> 8 Clusters

# num_clusters .metric        .estimator  mean     n std_err .config             
#            2 silhouette_avg standard   0.364    15 0.00872 Preprocessor1_Model1
#            4 silhouette_avg standard   0.458    15 0.0114  Preprocessor1_Model2
#            6 silhouette_avg standard   0.548    15 0.00911 Preprocessor1_Model3
#            8 silhouette_avg standard   0.632    15 0.00870 Preprocessor1_Model4

# 2. Predictions using clustering
cluster_model = k_means(num_clusters = 8) %>%
  set_engine("ClusterR")
cluster_wflow = workflow() %>%
  add_model(cluster_model) %>%
  add_recipe(cluster_recipe)

cluster_fit = fit(cluster_wflow, data = train_freena)

cluster_fit %>% extract_cluster_assignment()
# cluster_fit %>% extract_centroids()

cur_set = train_lm %>% filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01")))
final_set = cur_set %>%
  bind_cols(predict(cluster_fit, new_data = cur_set))
# View(final_set %>% skim_without_charts())

# Rerun the code below, but replace mutate for each predictor
# new_tests   total_tests   positive_rate   total_vaccinations    people_vaccinated
# extreme_poverty

# NOTE: only extreme_poverty in final_set above
final_set = final_set %>%
  group_by(.pred_cluster) %>%
  mutate(extreme_poverty = ifelse(
    is.na(extreme_poverty),
    median(extreme_poverty, na.rm = TRUE),
    extreme_poverty)) %>%
  ungroup()

# This is the training set with missingness imputed between Feb 2021 to March 2022
# Merge this dataset with the training set outside the above date range to
# get the final training set
temp_set = train_lm  %>% filter(date < as.Date("2021-02-01") | date > as.Date("2022-03-01"))
final_train = temp_set %>% bind_rows(final_set %>% select(-c(.pred_cluster)))

# Do the same for the testing set
temp_set = test_lm  %>% filter(date < as.Date("2021-02-01") | date > as.Date("2022-03-01"))
final_test = temp_set %>% bind_rows(final_set %>% select(-c(.pred_cluster)))

# Should we save?

ggplot(final_set, aes(date, extreme_poverty)) +
  geom_point(aes(color = location), alpha = 0.1)






