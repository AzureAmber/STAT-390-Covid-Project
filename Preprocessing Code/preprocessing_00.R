library(tidyverse)
data <- read_csv("data/raw_data/owid-covid-data.csv")
missingness <- naniar::miss_var_summary(data)

# preprocessing for just the usa data

data %>% 
  filter(iso_code == "USA") %>% 
  naniar::miss_var_summary() %>% 
  view()


usa_data <- data %>% 
  filter(iso_code == "USA") %>% 
  select(-weekly_icu_admissions, -weekly_icu_admissions_per_million, -handwashing_facilities)


usa_data %>% 
  filter(!is.na(excess_mortality)) # appears to be reported once per week, so we fill up

temp <- usa_data %>%
  fill(excess_mortality,.direction = "updown") %>% 
  select(-excess_mortality_cumulative, -excess_mortality_cumulative_absolute, -excess_mortality_cumulative_per_million)

temp %>%
  select(which(map_lgl(., ~ var(.x) == 0))) %>% 
  view() # rows with no info

temp <- temp %>%
  select(-which(map_lgl(., ~ var(.x) == 0)))
# the two indices with na just means it was not availbale before a certain time, na is fine


# booster, vaccine, tests, and hospital related info

temp %>% 
  filter(!is.na(total_boosters)) %>% 
  select(total_boosters) # after 2021 2 4 , which is when first booster was administered

#similar situation with tests and vaccines, na until they are availbale so we code na into 0

temp <- temp %>% 
  select(-total_boosters_per_hundred, -new_tests_smoothed, - new_tests_smoothed_per_thousand, -total_tests_per_thousand,
         -new_tests_per_thousand, -new_vaccinations_smoothed, - new_vaccinations_smoothed_per_million, -new_people_vaccinated_smoothed_per_hundred,
         -total_vaccinations_per_hundred, -people_vaccinated_per_hundred, -people_fully_vaccinated_per_hundred,
         -weekly_hosp_admissions_per_million, -icu_patients_per_million, -hosp_patients_per_million, -new_deaths_smoothed_per_million,
         -new_cases_smoothed_per_million, -new_deaths_per_million, -new_cases_per_million, -total_deaths_per_million,
         -total_cases_per_million, -tests_units)

temp %>% 
  filter(is.na(reproduction_rate))

temp %>% 
  naniar::miss_var_summary() %>% 
  view()

index <- temp %>% 
  select(reproduction_rate, stringency_index)

no_index <- temp %>% 
  select(-reproduction_rate, -stringency_index)
no_index <- no_index %>% 
  mutate_all(~replace_na(.,0))

final <- no_index %>% 
  cbind(index)

final  %>% 
  naniar::miss_var_summary()

