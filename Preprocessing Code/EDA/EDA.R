# Load packages 
library(tidyverse)
library(skimr)
library(tidyr)
library(gridExtra)
library(tseries)
library(zoo)
library(stats)
library(patchwork)


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

data_sorted <- data_cur |> 
  arrange(date) |> 
  select(date, new_cases) |> 
  na.omit()

# TEMPORAL VISUALIZATION ----
# 1. basic new_cases from 2020 to 2023
ggplot(data_sorted, aes(x = date, y = new_cases)) + 
  geom_line() + 
  labs(x = "Date", y = "New Cases", title = "Daily New Covid Cases") + 
  theme_bw()

# 2. looking at each of the 4 years 
ggplot(data_sorted, aes(x = date, y = new_cases)) +
  geom_line() +
  facet_wrap(~ format(date, "%Y"), scales = "free_x") +  
  labs(x = "Date", y = "New Cases") + 
  theme_bw()

# STATIONARITY ----
data_ts <- ts(data_sorted$new_cases, start = c(2020, 1), frequency = 365)
adf.test(data_ts)
# Augmented Dickey-Fuller Test
# 
# data:  data_ts
# Dickey-Fuller = -12.743, Lag order = 31, p-value = 0.01
# alternative hypothesis: stationary


# CORRELATION ANALYSIS ----
# Autocorrelation
acf(data_ts, lag.max = 365, main = "ACF For New Cases", ylim = c(-1, 1))


# Partial Autocorrelation
pacf(data_ts, main = "PACF for New Cases", ylim = c(-1, 1))

# SEASONAL DECOMPOSITION ----
# apply stl decomp
decomposed <- stl(data_ts, s.window = "periodic")
plot(decomposed)

# UNIVARIATE ANALYSIS ----

## response variable `new_cases`

# check for missingness
if (any(is.na(data_cur$new_cases))) {
  
  missing_response <- data_cur[is.na(data_cur$new_cases), c("continent", "location", "date")]
  
  missing_response <- missing_response[order(missing_response$date), ]
  
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

vars_to_plot <- names(x)[sapply(x, is.numeric) & !sapply(x, is.factor)]
vars_to_plot <- vars_to_plot[vars_to_plot != "new_cases"]

plot_list <- lapply(vars_to_plot, function(var_name) {
  ggplot(x, aes_string(x=var_name)) + 
    geom_histogram(fill="skyblue", color="black", alpha=0.7, na.rm = TRUE) +
    labs(x=var_name, y="Count")+
    theme_bw()
})

combined_plot <- do.call(grid.arrange, c(plot_list, ncol=5))

ggsave("EDA/predictors_histograms.png", combined_plot, width=20, height=15)


# BIVARIATE ANALYSIS ----

# plotting each numeric predictor against new_cases
plot_list2 <- list()

# loop through each variable and create a scatter plot
for (var_name in vars_to_plot) {
  if (var_name %in% colnames(x)) {  # Check if the variable exists in the dataset
    plot <- ggplot(x, aes(x = .data[[var_name]], y = new_cases)) +
      geom_point(color = "skyblue", alpha = 0.7, na.rm = TRUE) +
      labs(x = var_name, y = "new_cases") +
      theme_bw()
    plot_list2[[var_name]] <- plot  # Store the plot in the list
  } else {
    warning(paste("Variable", var_name, "not found in the dataset. Skipping."))
  }
}


bivariate_plots <- do.call(grid.arrange, c(plot_list2, ncol=5))
ggsave("EDA/predictors_vs_newcases.png", bivariate_plots, width = 20, height = 15)

# look closer for total_cases, new_deaths, reproduction_rate

# Multivariate Analysis 

# total cases
ggplot(data_cur, aes(x = total_cases, y = new_cases)) + 
  geom_point(color = "skyblue", alpha = 0.7, na.rm = TRUE) + 
  facet_wrap(~ location) + 
  theme_bw()

  # china specifically 
ggplot(data_cur |> filter(location == "China"), aes(x = total_cases, y = new_cases)) + 
  geom_point(color = "skyblue", alpha = 0.7, na.rm = TRUE) + 
  geom_smooth() + 
  facet_wrap(~ location) + 
  theme_bw()

# new deaths 
ggplot(data_cur , aes(x = new_deaths, y = new_cases)) + 
  geom_point(color = "skyblue", alpha = 0.7, na.rm = TRUE) + 
  facet_wrap(~ location) + 
  theme_bw()

  # china specifically
ggplot(data_cur |> filter(location == "China"), aes(x = new_deaths, y = new_cases)) + 
  geom_point(color = "skyblue", alpha = 0.7, na.rm = TRUE) + 
  geom_smooth() + 
  facet_wrap(~ location) + 
  theme_bw()

# reproduction rate
ggplot(data_cur, aes(x = reproduction_rate, y = new_cases)) + 
  geom_point(color = "skyblue", alpha = 0.7, na.rm = TRUE) + 
  facet_wrap(~ location) + 
  theme_bw()

# Correlation plot 
cor_matrix <- cor(x |> select(all_of(vars_to_plot)) |> na.omit())
corrplot::corrplot(cor_matrix, method = "color", diag = TRUE, 
         tl.col = "black", tl.srt = 45, tl.cex = 0.7) 

