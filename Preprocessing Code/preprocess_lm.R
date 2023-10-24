library(tidyverse)
library(skimr)
library(lubridate)
library(cluster)
library(factoextra)

# linear models = remove observations
# tree based models = give large values
# neural networks = create boolean indicator variable 

data = read_csv('data/raw_data/owid-covid-data.csv')

g20 = c('Argentina', 'Australia', 'Canada', 'China', 'France', 'Germany',
        'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
        'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')
g24 = c('Argentina', 'China', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
        'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')
data_cur = data %>%
  filter(location %in% c(g20, g24)) %>%
  mutate(G20 = location %in% g20, G24 = location %in% g24)



# remove variables with large missingness OR collinearity
results = data_cur %>% skim_without_charts()
names_filt = results$skim_variable[results$complete_rate < 0.7]
names_col = c('iso_code', 'median_age', 'aged_65_older', 'aged_70_older',
              'human_development_index', 'new_cases_smoothed', 'new_deaths_smoothed',
              'new_deaths_smoothed_per_million', 'new_cases_smoothed_per_million')
data_lm = data_cur %>% select(-any_of(c(names_filt, names_col)))
# View(cor(data_tree %>% select(-c(continent, location, date, G20, G24)), use = "complete.obs"))
# SPLIT INTO TRAINING AND TESTING SET HERE FOR THE ABOVE DATAS

train_lm  <- data_lm |> arrange(date) %>% filter(date < as.Date("2023-01-01"))
test_lm <- data_lm |> arrange(date) %>% filter(date >= as.Date("2023-01-01"))

# write_rds(train_lm, 'data/processed_data/train_lm.rds')
# write_rds(test_lm, 'data/processed_data/test_lm.rds')




# Imputation for random missingness
# Either impute using process below   OR    impute by clustering
# - new_deaths impute with 0, total deaths by last non-zero value
# - total_cases by last non-zero value, new_cases by change in total_cases
# - reproduction_rate by last non-zero value
# - extreme_poverty by median extreme_poverty value by continent
# - rest of other predictors by last non-zero value

#   ***** REPLACE data_lm here with the training set *****
data_lm2 = train_lm  %>%
  mutate(
    new_deaths = ifelse(is.na(new_deaths), 0, new_deaths),
    new_deaths_per_million = ifelse(is.na(new_deaths_per_million), 0, new_deaths_per_million)
  )
data_lm2 = data_lm2 %>%
  group_by(location) %>%
  fill(total_deaths, .direction = "downup") %>%
  ungroup()
data_lm2 = data_lm2 %>%
  mutate(
    total_deaths_per_million = ifelse(is.na(total_deaths_per_million), total_deaths / population * 1e6, total_deaths_per_million)
  )
data_lm2 = data_lm2 %>%
  group_by(location) %>%
  fill(total_cases, .direction = "downup") %>%
  ungroup()
data_lm2 = data_lm2 %>%
  mutate(
    total_cases_per_million = ifelse(is.na(total_cases_per_million), total_cases / population * 1e6, total_cases_per_million)
  )
data_lm2 = data_lm2 %>%
  group_by(location) %>%
  mutate(
    new_cases = ifelse(is.na(new_cases), total_cases - lag(total_cases, default = 0), new_cases)
  ) %>%
  ungroup()
data_lm2 = data_lm2 %>%
  mutate(
    new_cases_per_million = ifelse(is.na(new_cases_per_million), new_cases / population * 1e6, new_cases_per_million)
  )
data_lm2 = data_lm2 %>%
  group_by(location) %>%
  fill(reproduction_rate, .direction = "downup") %>%
  ungroup()
data_lm2 = data_lm2 %>% 
  group_by(continent) %>% 
  mutate(extreme_poverty = ifelse(is.na(extreme_poverty), median(extreme_poverty, na.rm = TRUE), extreme_poverty)) %>% 
  ungroup()
data_lm2 = data_lm2 %>% 
  group_by(location) %>% 
  mutate(stringency_index = ifelse(is.na(stringency_index), median(stringency_index, na.rm = TRUE), stringency_index)) %>% 
  ungroup()


# View(data_lm2 %>% skim_without_charts())
# View(cor(data_lm2 %>% select(-c(continent, location, date, G20, G24))))



# **CLUSTERING METHOD**
# Reference: https://www.statology.org/k-means-clustering-in-r/

# 1. Set seed
set.seed(390)

#2. Find optimal # of clusters using gap statistic
data_lm3 <- data_lm2 |> 
    select_if(~ all(!is.na(.))) |>    # Select columns with no NA values
    select_if(is.numeric) |>          # Select numeric columns
    scale()                           # Scale the columns

gap_stat <- clusGap(data_lm3,
                    FUN = kmeans,
                    nstart = 25,
                    K.max = 10,
                    B = 50)

# Plot number of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

#2.1 Can also find optimal K with this method (takes longer to run...)
# fviz_nbclust(data_lm3, kmeans, method = "wss")








