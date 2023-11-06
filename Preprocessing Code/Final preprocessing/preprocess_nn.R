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
data_nn = data_cur %>% select(-any_of(c(names_filtn, names_col)))
# View(cor(data_tree %>% select(-c(continent, location, date, G20, G24)), use = "complete.obs"))



# Imputation for large column missingness: Neural Network = create boolean indicators
# TRUE if the entry has value and False if the entry is missing
data_nn2 = data_nn %>%
  mutate(
    total_tests_b = ifelse(is.na(total_tests), FALSE, TRUE),
    new_tests_b = ifelse(is.na(new_tests), FALSE, TRUE),
    positive_rate_b = ifelse(is.na(positive_rate), FALSE, TRUE),
    total_vaccinations_b = ifelse(is.na(total_vaccinations), FALSE, TRUE),
    extreme_poverty_b = ifelse(is.na(extreme_poverty), FALSE, TRUE),
    stringency_index_b = ifelse(is.na(stringency_index), FALSE, TRUE)
  )
# SPLIT INTO TRAINING AND TESTING SET HERE FOR THE ABOVE DATAS

train_nn  <- data_nn2 |> arrange(date) %>% filter(date < as.Date("2023-01-01"))
test_nn <- data_nn2 |> arrange(date) %>% filter(date >= as.Date("2023-01-01"))

# write_rds(train_nn, 'data/processed_data/train_nn.rds')
# write_rds(test_nn, 'data/processed_data/test_nn.rds')



# Imputation for random missingness using clustering
# refer to file clustering_nn.R



# View(data_nn3 %>% skim_without_charts())
# View(cor(data_nn3 %>% select(-c(continent, location, date, G20, G24))))







