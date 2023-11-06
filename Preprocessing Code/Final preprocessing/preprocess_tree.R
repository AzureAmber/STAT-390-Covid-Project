library(tidyverse)
library(skimr)
library(lubridate)

# linear models = remove predictors
# tree based models = give large values
# neural networks = create boolean indicator variable 

data = read_csv('data/raw_data/owid-covid-data.csv')

g20 = c('Argentina', 'Australia', 'Canada', 'France', 'Germany',
        'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
        'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')
g24 = c('Argentina', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
        'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')
data_cur = data %>%
  filter(location %in% c(g20, g24)) %>%
  mutate(G20 = location %in% g20,
         G24 = location %in% g24,
         month = as.factor(month(date)), 
         day_of_week = weekdays(date))



# remove variables with large missingness OR collinearity
results = data_cur %>% skim_without_charts()
names_filt = results$skim_variable[results$complete_rate < 0.7]
names_filtn = setdiff(names_filt,
                      c('total_tests', 'new_tests',
                        'positive_rate', 'total_vaccinations'))
names_col = c('iso_code', 'median_age', 'aged_65_older', 'aged_70_older',
              'human_development_index', 'new_cases_smoothed', 'new_deaths_smoothed',
              'new_deaths_smoothed_per_million', 'new_cases_smoothed_per_million')
data_tree = data_cur %>% select(-any_of(c(names_filtn, names_col)))
# View(cor(data_tree %>% select(-c(continent, location, date, G20, G24)), use = "complete.obs"))



#   ***** REPLACE data_tree2 here with the training set *****
# Imputation for large column missingness: Tree based = Large Value
v = 1e15
data_tree2 = data_tree %>%
  mutate(
    total_tests = ifelse(is.na(total_tests) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                         v, total_tests),
    new_tests = ifelse(is.na(new_tests) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                       v, new_tests),
    positive_rate = ifelse(is.na(positive_rate) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                           v, positive_rate),
    total_vaccinations = ifelse(is.na(total_vaccinations) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                                v, total_vaccinations)
  )
# SPLIT INTO TRAINING AND TESTING SET HERE FOR THE ABOVE DATAS

train_tree <- data_tree2 |> arrange(date) %>% filter(date < as.Date("2023-01-01"))
test_tree <- data_tree2 |> arrange(date) %>% filter(date >= as.Date("2023-01-01"))

# write_rds(train_tree, 'data/processed_data/train_tree.rds')
# write_rds(test_tree, 'data/processed_data/test_tree.rds')

# View(train_tree %>%
#        filter(between(date, as.Date("2021-02-01"), as.Date("2022-03-01"))) %>%
#        skim_without_charts())



# Imputation for random missingness using clustering
# refer to file clustering_tree.R




