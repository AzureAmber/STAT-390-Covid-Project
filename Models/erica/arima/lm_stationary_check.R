library(tidyverse)
library(tidymodels)
library(modeltime)
library(doParallel)
library(dplyr)
library(tseries)


# Setup parallel processing
# detectCores(logical = FALSE)
cores.cluster <- makePSOCKcluster(10)
registerDoParallel(cores.cluster)


# Read in data
train_lm <- read_rds('data/avg_final_data/final_train_lm.rds')
test_lm <- read_rds('data/avg_final_data/final_test_lm.rds')

#save different country in separate df


locations <- unique(train_lm$location)

for(loc in locations){
  print(loc)
  location_data <- train_lm %>% filter(location == loc)
  location_name <- make.names(loc)
  assign(location_name, location_data, envir = .GlobalEnv)
}


#check for stationary for each country

adf_results_list <- list()

for(loc in locations){
  dataframe_name <- make.names(loc)
  
  df<-get(dataframe_name)
  
  adf_test <- adf.test(df$new_cases, alternative = 'stationary')
  
  adf_results_list[[loc]] <- list(
    ADF_Statistic = adf_test$statistic,
    P_Value = adf_test$p.value,
    Stationary = adf_test$p.value < 0.05
  )
}

# check adf_results_list
# non-stationary: Japan, Sri Lanka
# the rest 21 countries are stationary


## check distribution of all 6 non-stationary countries
countries_of_interest <- c("Japan", "Sri Lanka")

print(is.data.frame(train_lm))
print("location" %in% names(train_lm))

filtered_train_lm <- train_lm[train_lm$location %in% countries_of_interest,]

filtered_train_lm %>% 
  ggplot(aes(x=new_cases))+
  geom_histogram(bins=30, fill='skyblue', alpha= 0.7)+
  facet_wrap(~location, scales = "free_y")+
  theme_bw()+
  labs(title = "Distribution of New Cases by Non-stationary Countries",
       x = "New cases", y = "Frequency")+
  theme(legend.position = "none")


### choose USA for stationary rep, choose Japan for non-stationary rep

