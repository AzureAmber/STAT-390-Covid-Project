library(tidyverse)
library(skimr)
library(lubridate)
library(cluster)
library(factoextra)

# linear models = remove observations
# tree based models = give large values
# neural networks = create boolean indicator variable 

data <- read_csv('data/raw_data/owid-covid-data.csv')

g20 <- c('Argentina', 'Australia', 'Canada', 'France', 'Germany',
        'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
        'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')
g24 <- c('Argentina', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
        'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')
data_cur <- data %>%
  filter(location %in% c(g20, g24)) %>%
  mutate(G20 = location %in% g20,
         G24 = location %in% g24,
         month = as.factor(month(date)), 
         day_of_week = weekdays(date)) 

data %>% 
  filter(location %in% c(g20, g24)) %>%
  mutate(G20 = location %in% g20,
         G24 = location %in% g24,
         month_name = month.abb[month(date)],
         month = month(date),
         year = as.factor(year(date)),
         day_of_week = weekdays(date)
  ) %>% 
  ggplot(aes( x= month, y = new_cases, color = year)) +
  geom_line()
  



# remove variables with large missingness OR collinearity
results = data_cur %>% skim_without_charts()
names_filt = results$skim_variable[results$complete_rate < 0.7]
names_col = c('iso_code', 'median_age', 'aged_65_older', 'aged_70_older',
              'human_development_index', 'new_cases_smoothed', 'new_deaths_smoothed',
              'new_deaths_smoothed_per_million', 'new_cases_smoothed_per_million')
data_lm = data_cur %>% select(-any_of(c(names_filt, names_col)))
# View(cor(data_tree %>% select(-c(continent, location, date, G20, G24)), use = "complete.obs"))
# SPLIT INTO TRAINING AND TESTING SET HERE FOR THE ABOVE DATAS

train_lm  <- data_lm |> arrange(date) %>% filter(date < as.Date("2023-01-01"))
test_lm <- data_lm |> arrange(date) %>% filter(date >= as.Date("2023-01-01"))

# write_rds(train_lm, 'data/processed_data/train_lm.rds')
# write_rds(test_lm, 'data/processed_data/test_lm.rds')



# outlier check using cook's distance > 0.5
model = lm(new_cases ~ location + new_deaths_per_million, data = data_lm)
plot(model, which = 4)
which(cooks.distance(model) > 0.5)
# There is only one outlier with large leverage
# new_deaths_per_million = Ecuador = 7324
# There isn't much outliers so don't change anything


# View(data_lm2 %>% skim_without_charts())
# View(cor(data_lm2 %>% select(-c(continent, location, date, G20, G24))))



# **CLUSTERING METHOD**
# Reference: https://www.statology.org/k-means-clustering-in-r/

# 1. Set seed
set.seed(390)

#2. Find optimal # of clusters using gap statistic
data_lm3 <- data_lm2 |> 
    select_if(~ all(!is.na(.))) |>    # Select columns with no NA values
    select_if(is.numeric) |>          # Select numeric columns
    scale()                           # Scale the columns

gap_stat <- clusGap(data_lm3,
                    FUN = kmeans,
                    nstart = 25,
                    K.max = 10,
                    B = 50)

# Plot number of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

#2.1 Can also find optimal K with this method (takes longer to run...)
# fviz_nbclust(data_lm3, kmeans, method = "wss")








