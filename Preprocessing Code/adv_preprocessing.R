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
        'India', 'Indonesia', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
        'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')
# Syria, Congo, Cote d\'Ivoire, Nigeria, Gabon, Guatemala, Syria, Trinidad, Venezuela, Peru are excluded
g24 = c('Algeria', 'Argentina', 'Brazil', 'China', 'Colombia', 'Ecuador', 'Egypt',
        'Ethiopia', 'Ghana', 'Haiti', 'India', 'Iran', 'Kenya', 'Lebanon', 'Mexico',
        'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')

data_cur = data %>% filter(location %in% c(g20, g24))



# remove variables with large missingness
results = data_cur %>% skim_without_charts()
names_filt = results$skim_variable[results$complete_rate < 0.7]

data_clean = data_cur %>% select(-any_of(names_filt))

# remove collinearity
# View(cor(data_clean %>% select(-any_of(c('iso_code', 'continent', 'location', 'date'))), use = 'complete.obs'))

names_col = c('iso_code', 'median_age', 'aged_65_older', 'aged_70_older', 'human_development_index', 'total_deaths',
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
data_clean3 = data_clean2 %>% filter(date < as.Date("2023-01-01"))

# View(data_clean3 %>% skim_without_charts())



