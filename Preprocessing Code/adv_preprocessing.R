library(tidyverse)
library(skimr)
library(lubridate)

data = read_csv('data/raw_data/owid-covid-data.csv')

# countries in dataset
# View(data %>% skim_without_charts())
# View(table(data$location))

# select data by the g20 or g24
# EU is excluded is many of its data is missing
g20 = c('Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany',
        'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
        'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')
# Syria, Congo, Cote d\'Ivoire, Nigeria, Gabon, Guatemala, Syria, Trinidad, Venezuela, Peru are excluded
g24 = c('Argentina', 'Brazil', 'China', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
        'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')



# data_cur = data %>% filter(location %in% c(g20, g24))
data_clean = data %>% filter(location %in% c(g20, g24))



# remove variables with large missingness
# results = data_cur %>% skim_without_charts()
# names_filt = results$skim_variable[results$complete_rate < 0.7]
# 
# data_clean = data_cur %>% select(-any_of(names_filt))

# remove collinearity
# View(cor(data_clean %>% select(-any_of(c('iso_code', 'continent', 'location', 'date'))), use = 'complete.obs'))

names_col = c('iso_code', 'median_age', 'aged_65_older', 'aged_70_older', 'human_development_index',
              'new_cases_smoothed', 'new_deaths_smoothed', 'new_deaths_smoothed_per_million', 'new_cases_smoothed_per_million')

data_clean2 = data_clean %>% select(-any_of(names_col))

# View(cor(data_clean2 %>% select(-any_of(c('iso_code', 'continent', 'location', 'date'))), use = 'complete.obs'))




# missingness by rows
# View(data_clean2 %>% skim_without_charts())
sort(table(data_clean2$location[is.na(data_clean2$extreme_poverty)]))
# France, Germany, Japan, Lebanon, Philippines, Saudi Arabia is missing extreme_poverty

# sort(table(data_clean2$date[is.na(data_clean2$stringency_index)]))
# x = data_cur %>% filter(is.na(stringency_index))
# y = x %>% group_by(location) %>% summarise(v = n(), mi = min(date), mx = max(date))
# it appears stringency index is missing starting at 2023
# *** Note *** This date can vary by country. Might need to adjust
data_clean3 = data_clean2 %>% filter(date < as.Date("2023-01-01"))

# View(data_clean3 %>% skim_without_charts())
# View(data_clean3[is.na(data_clean3$new_deaths),])





# random missingness
# Since the amount of missingness is few for new_deaths, impute with 0
data_clean4 = data_clean3 %>%
  mutate(
    new_deaths = ifelse(is.na(new_deaths), 0, new_deaths),
    new_deaths_per_million = ifelse(is.na(new_deaths_per_million), 0, new_deaths_per_million)
  )
# Replace missingness in total deaths by the last non-zero value
data_clean4 = data_clean4 %>%
  group_by(location) %>%
  fill(total_deaths, .direction = "downup") %>%
  ungroup()
data_clean4 = data_clean4 %>%
  mutate(
    total_deaths_per_million = ifelse(is.na(total_deaths_per_million), total_deaths / population * 1e6, total_deaths_per_million)
  )
# Replace missingness in total cases by the last non-zero value
data_clean4 = data_clean4 %>%
  group_by(location) %>%
  fill(total_cases, .direction = "downup") %>%
  ungroup()
data_clean4 = data_clean4 %>%
  mutate(
    total_cases_per_million = ifelse(is.na(total_cases_per_million), total_cases / population * 1e6, total_cases_per_million)
  )
# Replace missingness in new cases by the change in total cases
data_clean4 = data_clean4 %>%
  group_by(location) %>%
  mutate(
    new_cases = ifelse(is.na(new_cases), total_cases - lag(total_cases, default = 0), new_cases)
  ) %>%
  ungroup()
data_clean4 = data_clean4 %>%
  mutate(
    new_cases_per_million = ifelse(is.na(new_cases_per_million), new_cases / population * 1e6, new_cases_per_million)
  )
# Replace missingness in reproduction rate by the last non-zero value
data_clean4 = data_clean4 %>%
  group_by(location) %>%
  fill(reproduction_rate, .direction = "downup") %>%
  ungroup()

# View(data_clean4 %>% skim_without_charts())





# create additional predictors
data_clean4 = data_clean4 %>%
  mutate(
    G20 = location %in% g20,
    G24 = location %in% g24
  )




# country missingness

data_clean5 = data_clean4 %>% 
  group_by(continent) %>% 
  mutate(extreme_poverty = ifelse(is.na(extreme_poverty), median(extreme_poverty, na.rm = TRUE), extreme_poverty)) %>% 
  ungroup() 

skimr::skim(data_clean5)



# additional column missingess
# linear = dont use impute 0
# tree based = give large value
# NN = binary


data_clean6 = data_clean5 %>%
  group_by(location) %>%
  mutate(
    tests_units = ifelse(is.na(tests_units) & date < as.Date("2021-02-01"), 0, tests_units),
    icu_patients = ifelse(is.na(icu_patients) & date < as.Date("2021-02-01"), 0, icu_patients),
    hosp_patients = ifelse(is.na(hosp_patients) & date < as.Date("2021-02-01"), 0, hosp_patients),
    total_tests = ifelse(is.na(total_tests) & date < as.Date("2021-02-01"), 0, total_tests),
    new_tests = ifelse(is.na(new_tests) & date < as.Date("2021-02-01"), 0, new_tests),
    positive_rate = ifelse(is.na(positive_rate) & date < as.Date("2021-02-01"), 0, positive_rate),
    tests_per_case = ifelse(is.na(tests_per_case) & date < as.Date("2021-02-01"), 0, tests_per_case),
    total_vaccinations = ifelse(is.na(total_vaccinations) & date < as.Date("2021-02-01"), 0, total_vaccinations),
    people_vaccinated = ifelse(is.na(people_vaccinated) & date < as.Date("2021-02-01"), 0, people_vaccinated),
    new_vaccinations = ifelse(is.na(new_vaccinations) & date < as.Date("2021-02-01"), 0, new_vaccinations)
  ) %>%
  ungroup()

View(data_clean6 %>% skim_without_charts())
