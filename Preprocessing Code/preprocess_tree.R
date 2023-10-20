library(tidyverse)
library(skimr)
library(lubridate)

# linear models = remove predictors
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
names_filtn = setdiff(names_filt,
                      c('icu_patients', 'hosp_patients', 'total_tests', 'new_tests',
                        'positive_rate', 'total_vaccinations', 'people_vaccinated'))
names_col = c('iso_code', 'median_age', 'aged_65_older', 'aged_70_older',
              'human_development_index', 'new_cases_smoothed', 'new_deaths_smoothed',
              'new_deaths_smoothed_per_million', 'new_cases_smoothed_per_million')
data_tree = data_cur %>% select(-any_of(c(names_filtn, names_col)))
# View(cor(data_tree %>% select(-c(continent, location, date, G20, G24)), use = "complete.obs"))
# SPLIT INTO TRAINING AND TESTING SET HERE FOR THE ABOVE DATAS



# Imputation for large column missingness: Tree based = Large Value
v = 1e15
data_tree2 = data_tree %>%
  mutate(
    icu_patients = ifelse(is.na(icu_patients) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                          v, icu_patients),
    hosp_patients = ifelse(is.na(hosp_patients) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                           v, hosp_patients),
    total_tests = ifelse(is.na(total_tests) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                         v, total_tests),
    new_tests = ifelse(is.na(new_tests) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                       v, new_tests),
    positive_rate = ifelse(is.na(positive_rate) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                           v, positive_rate),
    total_vaccinations = ifelse(is.na(total_vaccinations) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                                v, total_vaccinations),
    people_vaccinated = ifelse(is.na(people_vaccinated) & (date < as.Date("2021-02-01") | date > as.Date("2022-03-01")),
                               v, people_vaccinated)
  )



# Imputation for random missingness
# Either impute using process below   OR    impute by clustering
# - new_deaths impute with 0, total deaths by last non-zero value
# - total_cases by last non-zero value, new_cases by change in total_cases
# - reproduction_rate by last non-zero value
# - extreme_poverty by median extreme_poverty value by continent
# - rest of other predictors by last non-zero value
data_tree3 = data_tree2 %>%
  mutate(
    new_deaths = ifelse(is.na(new_deaths), 0, new_deaths),
    new_deaths_per_million = ifelse(is.na(new_deaths_per_million), 0, new_deaths_per_million)
  )
data_tree3 = data_tree3 %>%
  group_by(location) %>%
  fill(total_deaths, .direction = "downup") %>%
  ungroup()
data_tree3 = data_tree3 %>%
  mutate(
    total_deaths_per_million = ifelse(is.na(total_deaths_per_million), total_deaths / population * 1e6, total_deaths_per_million)
  )
data_tree3 = data_tree3 %>%
  group_by(location) %>%
  fill(total_cases, .direction = "downup") %>%
  ungroup()
data_tree3 = data_tree3 %>%
  mutate(
    total_cases_per_million = ifelse(is.na(total_cases_per_million), total_cases / population * 1e6, total_cases_per_million)
  )
data_tree3 = data_tree3 %>%
  group_by(location) %>%
  mutate(
    new_cases = ifelse(is.na(new_cases), total_cases - lag(total_cases, default = 0), new_cases)
  ) %>%
  ungroup()
data_tree3 = data_tree3 %>%
  mutate(
    new_cases_per_million = ifelse(is.na(new_cases_per_million), new_cases / population * 1e6, new_cases_per_million)
  )
data_tree3 = data_tree3 %>%
  group_by(location) %>%
  fill(reproduction_rate, .direction = "downup") %>%
  ungroup()
data_tree3 = data_tree3 %>% 
  group_by(continent) %>% 
  mutate(extreme_poverty = ifelse(is.na(extreme_poverty), median(extreme_poverty, na.rm = TRUE), extreme_poverty)) %>% 
  ungroup()
data_tree3 = data_tree3 %>% 
  group_by(location) %>% 
  mutate(stringency_index = ifelse(is.na(stringency_index), median(stringency_index, na.rm = TRUE), stringency_index)) %>% 
  ungroup()


data_tree3 = data_tree3 %>%
  group_by(location) %>%
  fill(icu_patients, .direction = "downup") %>%
  fill(hosp_patients, .direction = "downup") %>%
  fill(total_tests, .direction = "downup") %>%
  fill(positive_rate, .direction = "downup") %>%
  fill(total_vaccinations, .direction = "downup") %>%
  fill(people_vaccinated, .direction = "downup") %>%
  mutate(new_tests = ifelse(is.na(new_tests), 0, new_tests)) %>%
  ungroup()



# View(data_tree3 %>% skim_without_charts())
# View(cor(data_tree3 %>% select(-c(continent, location, date, G20, G24))))







