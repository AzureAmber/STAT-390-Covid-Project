# monthly average
library(tseries)
library(tidyverse)
library(skimr)
library(lubridate)

data = read_csv('data/raw_data/owid-covid-data.csv')

g20 = c('Argentina', 'Australia', 'Canada', 'China', 'France', 'Germany',
        'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
        'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')
g24 = c('Argentina', 'China', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
        'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')
data_cur = data %>%
  filter(location %in% c(g20, g24)) %>%
  mutate(G20 = location %in% g20, G24 = location %in% g24)


xx = data_cur %>%
  filter(location == "United States", date < as.Date('2023-05-01')) %>%
  select(new_cases, date)
xxx = xx %>%
  group_by(date = floor_date(date, "month")) %>%
  summarise(avg_new_cases = mean(new_cases, na.rm = TRUE))
ggplot(xxx, aes(date, avg_new_cases)) +
  geom_line() +
  scale_x_date(breaks = "2 month") +
  theme(axis.text.x = element_text(angle = 90))

y = ts(data = xxx %>% select(avg_new_cases), start = 1, frequency = 12)
acf(y, 40)
acf(y, 40, type = "partial")
adf.test(y)
z = decompose(y, type = "multiplicative")
z = decompose(y, type = "additive")
plot(z)


# new_cases monthly average Countries

# zz = data_cur %>%
#   filter(date < as.Date('2023-05-01')) %>%
#   select(new_cases, date, location, G20, G24)
# ggplot(zz %>% filter(G20), aes(date, new_cases)) +
#   geom_line() +
#   facet_wrap(~location, scales = "free_y")
# ggplot(zz %>% filter(G24), aes(date, new_cases)) +
#   geom_line() +
#   facet_wrap(~location, scales = "free_y")

thing = function(x) {
  xx = x %>% select(new_cases, date)
  xxx = xx %>%
    group_by(date = floor_date(date, "month")) %>%
    summarise(avg_new_cases = mean(new_cases, na.rm = TRUE))
  ts(data = xxx %>% select(avg_new_cases), start = 1, frequency = 12)
}

z = data_cur %>%
  filter(date < as.Date('2023-05-01')) %>%
  select(new_cases, date, location)
country = split(z, z$location)

k = sapply(country, thing)

country_acf = apply(k, 1, acf, lag.max = 40, plot = FALSE)
names(country_acf) = names(country)

country_pacf = apply(k, 1, acf, lag.max = 40, type = 'partial', plot = FALSE)
names(country_pacf) = names(country)



get_acf = function(x) {
  as.vector(x[["acf"]])
}

m = as.data.frame(sapply(country_acf, get_acf)[,1:24]) %>%
  mutate(lag = seq(0, 23, 1)) %>%
  pivot_longer(everything() & !lag, names_to = "country", values_to = "acf") %>%
  filter(lag > 0)
n = as.data.frame(sapply(country_pacf, get_acf)[,1:24]) %>%
  mutate(lag = seq(1, 23, 1)) %>%
  pivot_longer(everything() & !lag, names_to = "country", values_to = "acf")


ciz = (-qnorm((1-0.95)/2)/sqrt(1214-3))
cir = (exp(2*ciz)-1)/(exp(2*ciz)+1)

ggplot(m, aes(lag, 0)) +
  geom_segment(aes(xend = lag, yend = acf), color = 'blue') +
  geom_hline(yintercept = 0, color = 'black') +
  geom_hline(yintercept = cir, color = 'red', linetype = 'dashed') +
  geom_hline(yintercept = -1*cir, color = 'red', linetype = 'dashed') +
  facet_wrap(~country, scales = "free_y") +
  labs(title = "ACF of Average Monthly New Cases per Country", y = 'ACF')

ggplot(n, aes(lag, 0)) +
  geom_segment(aes(xend = lag, yend = acf), color = 'blue') +
  geom_hline(yintercept = 0, color = 'black') +
  geom_hline(yintercept = cir, color = 'red', linetype = 'dashed') +
  geom_hline(yintercept = -1*cir, color = 'red', linetype = 'dashed') +
  facet_wrap(~country, scales = "free_y") +
  labs(title = "PACF of Average Monthly New Cases per Country", y = 'PACF')











library(tseries)
data = tibble(
  country = numeric(23),
  adf = numeric(23),
  adf_pval = numeric(23),
  adf_state = numeric(23)
)
country_names = sort(unique(complete_lm$location))
for (i in 1:23) {
  dat = complete_lm %>% filter(location == country_names[i]) %>%
    arrange(date)
  x = ts(dat$new_cases, frequency = 7)
  y = adf.test(x)
  data$country[i] = country_names[i]
  data$adf[i] = y$statistic
  data$adf_pval[i] = y$p.value
  data$adf_state[i] = ifelse(y$p.value <= 0.05, "Stationary", "Non-Stationary")
}













