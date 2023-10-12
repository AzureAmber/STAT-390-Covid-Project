source('Preprocessing Code/adv_preprocessing.R')


x = data_cur %>% select(date, location, icu_patients)
x = x %>%
  group_by(location) %>%
  mutate(
    mi = ifelse(is.na(icu_patients), 0, 1),
    miavg = cummean(mi)
  )


ggplot(x, aes(date, miavg)) +
  geom_point(aes(color = location))























