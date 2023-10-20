source('Preprocessing Code/adv_preprocessing.R')


x <- data_cur %>% select(date, location, icu_patients)
x <- x %>%
  group_by(location) %>%
  mutate(
    mi = ifelse(is.na(icu_patients), 0, 1),
    miavg = cummean(mi)
  )


ggplot(x, aes(date, miavg)) +
  geom_point(aes(color = location))


#vax/hospital/icu and target variable relationship? -> if feature is significant
data_cur %>% 
  ggplot(aes(x = total_vaccinations, y = new_cases)) +
  geom_point()
# seems like there is a relationship to some extent, the middle valley is due to missingness in vax
# 
# data_cur %>% 
#   ggplot(aes(x = hosp_patients, y = new_cases)) +
#   geom_point()


#if so, inner join 2023 data back
data_cur %>% 
  filter(!is.na(total_vaccinations))

first_report <- data_cur %>% 
  filter(!is.na(total_vaccinations)) %>% 
  group_by(iso_code) %>% 
  summarise(first_report = min(date))
data_cur_w_first_report <- data_cur %>% 
  full_join(first_report, by = "iso_code")



data_clean_vax <- data_cur_w_first_report %>% 
  mutate(total_vaccinations = ifelse(date < first_report,replace_na(total_vaccinations,0), total_vaccinations)) %>% 
  select(iso_code, total_vaccinations, date, first_report) %>% 
  filter(is.na(total_vaccinations))

#impute missingness before first vaccination came out as 0

#impute missingness in more recent data (assumption is they just stopped reporting) using comparable's data to do a regression analysis







