library(tidyr)

temp = rbind(train_lm, test_lm)

data = NULL




# replace weekly total by daily average
v = temp %>% filter(location == 'South Korea') %>%
  arrange(date)
x = v %>% filter(!between(date, as.Date('2023-6-15'), as.Date('2023-9-4')))
y = v %>% filter(between(date, as.Date('2023-6-15'), as.Date('2023-9-4'))) %>%
  arrange(date)
z = y %>% mutate(
  version_count = ifelse(new_cases == 0, 0, 1),
  version = ifelse(new_cases == 0, NA, cumsum(version_count))) %>%
  fill(version, .direction = 'up') %>%
  group_by(version) %>%
  mutate(new_cases = floor(mean(new_cases))) %>%
  ungroup() %>%
  select(-c(version, version_count))



p = rbind(x,z) %>% arrange(date)
data = rbind(data, p)

# data = rbind(data, v)



