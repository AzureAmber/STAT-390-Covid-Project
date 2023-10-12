# STAT 390 Project 
# Data Exploration ----

## Load packages -----
library(tidyverse)
library(corrplot)

source('Preprocessing Code/adv_preprocessing.R')


theme_set(theme_minimal())

cor_set <- data_clean4 %>% 
  select(new_cases, where(is.numeric))

correlation <- cor(cor_set[, -which(names(cor_set) == "new_cases")], as.numeric(cor_set$new_cases), use = "pairwise.complete.obs")

correlation |> 
  enframe() |> 
  arrange(desc(value))

# new_deaths, total_cases, population

ggplot(data_clean4, aes(new_deaths, new_cases)) + 
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  geom_point()

ggplot(data_clean4, aes(total_cases, new_cases)) + 
  geom_point()

ggplot(data_clean4, aes(population, new_cases)) + 
  geom_point()

# do we need to apply any transformation on new_cases?
