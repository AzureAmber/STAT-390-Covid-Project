# Load packages 
library(tidyverse)
library(skimr)
library(tidyr)
library(gridExtra)


#load data
data = read_csv('data/raw_data/owid-covid-data.csv')

g20 = c('Argentina', 'Australia', 'Canada', 'China', 'France', 'Germany',
        'India', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Russia',
        'Saudi Arabia', 'South Africa', 'Turkey', 'United Kingdom', 'United States')
g24 = c('Argentina', 'China', 'Colombia', 'Ecuador', 'Ethiopia', 'India',
        'Mexico', 'Morocco', 'Pakistan', 'Philippines', 'South Africa', 'Sri Lanka')
data_cur = data %>%
  filter(location %in% c(g20, g24)) %>%
  mutate(G20 = location %in% g20, G24 = location %in% g24)


## first check the distribution of response variable

# Check for missingness in 'new_cases'
if (any(is.na(data_cur$new_cases))) {
  
  # Extract rows where 'new_cases' is missing and select relevant columns
  missing_response <- data_cur[is.na(data_cur$new_cases), c("continent", "location", "date")]
  
  # Sort data by 'date' in ascending order
  missing_response <- missing_data[order(missing_data$date), ]
  
  # Print the table
  print(missing_response)
} else {
  cat("There are no missing values in 'new_cases'.\n")
}

## there are **161** missing response values, mainly at the beginning of the COVID outbreak
## before 2020/9 or more recently after 2023/5

# histogram of the response variable

data_cur %>% 
  ggplot(aes(x=new_cases)) + 
  geom_histogram(fill="skyblue", color="black", alpha=0.7) +
  labs(title="Histogram of New Cases", x="New Cases", y="Count")+
  theme_bw()+
  ggsave("EDA/response.png")

## the shape is heavily skewed to the right

# histogram for new_cases with log-transformed x-axis
data_cur %>% 
  ggplot(aes(x=new_cases)) + 
  geom_histogram(fill="skyblue", color="black", alpha=0.7, na.rm = TRUE) +
  scale_x_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  labs(title="Histogram of New Cases (Log Scale)", x="New Cases (Log Scale)", y="Count")+
  theme_bw() + 
  ggsave("EDA/log_response.png")

## looks much more normally distributed now!


# check the distribution of significant **predictor variables**

vars_to_plot <- names(train_lm)[sapply(train_lm, is.numeric) & !sapply(train_lm, is.factor)]

plot_list <- lapply(vars_to_plot, function(var_name) {
  ggplot(train_lm, aes_string(x=var_name)) + 
    geom_histogram(fill="skyblue", color="black", alpha=0.7, na.rm = TRUE) +
    labs(x=var_name, y="Frequency")+
    theme_bw()
})

combined_plot <- do.call(grid.arrange, c(plot_list, ncol=5))

ggsave("EDA/predictors_histograms.pdf", combined_plot, width=20, height=15)


