# NOTE: This is the cleaned up version of `adv_preprocessing.R` and `additional_preprocessing.R`
#       to make things clearer. 

# Load packages 
library(tidyverse)
library(skimr)
library(lubridate)

# Load data 
data <- read_csv('data/raw_data/owid-covid-data.csv')

# SUMMARY: 
    # `data_clean`: data for only G20 and G24 countries 
    # `data_clean2`: data w/ collinear vars removed (**USE FOR ARIMA**)
    # `data_clean3`: data before 2023 (ends 2022-12-31) bc of `stringency_index`
    # `data_clean4`: imputed missingness values with 0, last non-zero value, & change in value
    # `data_clean5`: imputed missingness for `extreme_poverty`
    # `data_clean5`: imputed missingness before 2021-02-01

# Variables and Their Data Availability Periods: 

    # 'tests_units': Data available from Jan 2021 to Apr 2022 for selected countries.
    # 'icu_patients': Data available from Jan 2021 to Apr 2022 for selected countries.
    # 'hosp_patients': Data available from Feb 2021 to June 2022 for selected countries.
    # 'weekly_icu_admissions': OMIT FROM DATA
    # 'total_tests': Data available from Nov 2020 to Apr 2022 for selected countries.
    # 'new_tests': Data available from Nov 2020 to Apr 2022 for selected countries.
    # 'positive_rate': Data available from Jan 2021 to Apr 2022 for selected countries.
    # 'tests_per_case': Data available from Jan 2021 to Apr 2022 for selected countries.
    # 'total_vaccinations': Data available from Jan 2021 to Mar 2022 for selected countries.
    # 'people_vaccinated': Data available from Jan 2021 to Mar 2022 for selected countries.
    # 'total_boosters': OMIT FROM DATA
    # 'new_vaccinations': Data available from Jul 2021 to June 2022 for selected countries.
    # 'handwashing_facilities': OMIT FROM DATA
    # 'excess_mortality': OMIT FROM DATA

# Note: Generally, most data spans from Jan 2021 to Mar/Apr 2022.
######################################################################################
# ---- Define G20 and G24 countries for analysis
g20 <- c('Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany',
         'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
         'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')

g24 <- c('Argentina', 'Brazil', 'China', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
         'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')

  # NOTE: EU is excluded as many of its data is missing
  # NOTE: Syria, Congo, Cote d\'Ivoire, Nigeria, Gabon, Guatemala, Syria, 
  #       Trinidad, Venezuela, Peru are excluded

# ---- Filter data for G20 and G24 countries
data_clean <- data %>% filter(location %in% c(g20, g24))


# ---- Identify collinear variables to remove
names_col <- c('iso_code', 'median_age', 'aged_65_older', 'aged_70_older', 'human_development_index',
               'new_cases_smoothed', 'new_deaths_smoothed', 'new_deaths_smoothed_per_million', 'new_cases_smoothed_per_million')
data_clean2 <- data_clean %>% select(-any_of(names_col))


# ---- Handle missing values
# Remove data after 2023 due to missingness in stringency index
data_clean3 <- data_clean2 %>% filter(date < as.Date("2023-01-01"))

# ---- Handle random missingness
# Impute new_deaths and new_deaths_per_million with 0 where they're missing
data_clean4 <- data_clean3 %>%
  mutate(
    new_deaths = ifelse(is.na(new_deaths), 0, new_deaths),
    new_deaths_per_million = ifelse(is.na(new_deaths_per_million), 0, new_deaths_per_million)
  )

# Fill missing values for several variables using the last non-zero value or the change in value
data_clean4 <- data_clean4 %>%
  group_by(location) %>%
  fill(total_deaths, total_cases, reproduction_rate, .direction = "downup") %>%
  ungroup() %>%
  mutate(
    total_deaths_per_million = ifelse(is.na(total_deaths_per_million), total_deaths / population * 1e6, total_deaths_per_million),
    total_cases_per_million = ifelse(is.na(total_cases_per_million), total_cases / population * 1e6, total_cases_per_million),
    new_cases = ifelse(is.na(new_cases), total_cases - lag(total_cases, default = 0), new_cases),
    new_cases_per_million = ifelse(is.na(new_cases_per_million), new_cases / population * 1e6, new_cases_per_million)
  )

# Generate new variables indicating G20 and G24 membership
data_clean4 <- data_clean4 %>%
  mutate(
    G20 = location %in% g20,
    G24 = location %in% g24
  )


# ---- Handle country-specific missingness
# Impute missing extreme_poverty values with the median value per continent
data_clean5 <- data_clean4 %>% 
  group_by(continent) %>% 
  mutate(extreme_poverty = ifelse(is.na(extreme_poverty), median(extreme_poverty, na.rm = TRUE), extreme_poverty)) %>% 
  ungroup() 



# ---- Handle missingness in additional columns before 2021-02-01
data_clean6 <- data_clean5 %>%
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

# Review the cleaned data
View(data_clean6 %>% skim_without_charts())

# FEATURE ENGINEERING ----
# Add new variables
data_clean7 <- data_clean6 |>
  mutate(month = as.factor(month(date)), 
         day_of_week = weekdays(date)) |> 
  select(date, month, day_of_week)
# need to add holidays ?

###########################################################################################
# Calculating Running Average Missingness for `excess_mortality` by Country 
x <- data_clean6 %>%
  select(date, location, excess_mortality) %>%
  group_by(location) %>%
  # Calculating a binary variable that indicates missingness for 'excess_mortality'
  mutate(
    mi = ifelse(is.na(excess_mortality), 0, 1),
    # Calculating the cumulative mean of the missingness indicator for visualization
    miavg = cummean(mi)
  )

# Plotting running average missingness over time
ggplot(x, aes(date, miavg)) +
  geom_point(aes(color = location)) +
  scale_x_date(date_breaks = "4 months") +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  theme(axis.text.x = element_text(angle = 90))

# Filtering countries with a running average missingness below 0.6 between Sep 2021 and May 2022
y = x %>% filter(between(date, as.Date("2021-09-01"), as.Date("2022-05-01")), miavg < 0.6)

# Displaying unique locations and their maximum missingness average during the specified time frame
noquote(y$location %>% unique())
View(y %>% group_by(location) %>% summarise(v = max(miavg)))


