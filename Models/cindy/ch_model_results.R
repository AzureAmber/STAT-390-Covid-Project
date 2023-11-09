# Results

# Get final model results 
library(tidyverse)
library(tidymodels)

tidymodels_prefer()

# # patterns selects all rda files
# result_files <- list.files("results/", "*.rda", full.names = TRUE)
# source('Models/cindy/5_xgboost.R')
# 
# # loads all the files into environment 
# # saves us from having to individually load each file 
# for(i in result_files){
#   load(i)
# }

########################################################################
# baseline/null model - CHECK!

null_model <- null_model(mode = "regression") %>% 
  set_engine("parsnip")


null_wflow <- workflow() %>% 
  add_model(null_model) %>% 
  add_recipe(btree_recipe)


null_fit <- null_wflow %>% 
  fit_resamples(resamples = data_folds,
                control = control_resamples(save_pred = TRUE))

null_fit %>% 
  collect_metrics()

# .metric .estimator   mean     n std_err .config             
# <chr>   <chr>       <dbl> <int>   <dbl> <chr>               
#   1 rmse    standard   40970.   205   3550. Preprocessor1_Model1
# 2 rsq     standard     NaN      0     NA  Preprocessor1_Model1
