# Load packages 
library(tidyverse)
library(skimr)
library(tidyr)
library(outliers)

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

#find outliers 

## ** USING OUTLIER PACKAGE **

numerical_vars <- sapply(data_cur, is.numeric)
numerical_data <- data_cur[, numerical_vars]

outliers_list <- list()

for (var_name in names(numerical_data)) {
  outlier_value <- outlier(numerical_data[[var_name]])
  
  outliers_list[[var_name]] <- outlier_value
}

for (var_name in names(outliers_list)) {
  cat(var_name, ":", outliers_list[[var_name]], "\n")
}


## ** ADD IQR METHOD in addition to OUTLIER **


# Exclude columns with 'new_cases' in their names
numerical_data <- numerical_data[, !grepl("new_cases", names(numerical_data))]

# Function to get outliers based on IQR method
find_iqr_outliers <- function(x) {
  x <- na.omit(x)  # Remove NAs
  
  quantiles <- quantile(x, c(0.25, 0.75))
  iqr <- IQR(x)
  
  lower_bound <- quantiles[1] - 1.5 * iqr
  upper_bound <- quantiles[2] + 1.5 * iqr
  
  outliers <- x[x < lower_bound | x > upper_bound]
  return(outliers)
}


variable_names <- c()
outlier_counts <- c()

# Loop through each numerical column
for (var_name in names(numerical_data)) {
  # Clean the column of NA values for the outlier function
  clean_column <- na.omit(numerical_data[[var_name]])
  
  # Using outlier package
  if(length(clean_column) > 0) {  
    outlier_pkg <- outlier(clean_column)
  } else {
    outlier_pkg <- numeric(0) 
  }
  
  # Using IQR method
  outliers_iqr <- find_iqr_outliers(numerical_data[[var_name]])
  
  # Combine and make unique
  all_outliers <- unique(c(outlier_pkg, outliers_iqr))
  
  # Store results
  variable_names <- c(variable_names, var_name)
  outlier_counts <- c(outlier_counts, length(all_outliers))
}

# Create a data frame with results and sort in descending order of outlier counts
result <- data.frame(variable = variable_names, count = outlier_counts)
result <- result[order(-result$count), ]

print(result)

# BOXPLOT FOR TOP 10 RESULTS

# Identify top 10 variables
top_10_vars <- head(result$variable, 10)

# Loop through each of the top 10 variables
for (var_name in top_10_vars) {
  # Draw the boxplot using ggplot2
  p <- ggplot(data_cur, aes(y = .data[[var_name]])) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", var_name), y = var_name)
  
  ggsave(filename = paste0("EDA/top10_outliers/", var_name, "_boxplot.png"), plot = p, width = 6, height = 4)
}


## find outliers for only siginificant predictor variables

numerical_vars <- sapply(x, is.numeric)
numerical_data <- x[, numerical_vars]


# Exclude columns with 'new_cases' in their names
numerical_data <- numerical_data[, !grepl("new_cases", names(numerical_data))]

# Function to get outliers based on IQR method
find_iqr_outliers <- function(x) {
  x <- na.omit(x)  # Remove NAs
  
  quantiles <- quantile(x, c(0.25, 0.75))
  iqr <- IQR(x)
  
  lower_bound <- quantiles[1] - 1.5 * iqr
  upper_bound <- quantiles[2] + 1.5 * iqr
  
  outliers <- x[x < lower_bound | x > upper_bound]
  return(outliers)
}


variable_names <- c()
outlier_counts <- c()

# Loop through each numerical column
for (var_name in names(numerical_data)) {
  # Clean the column of NA values for the outlier function
  clean_column <- na.omit(numerical_data[[var_name]])
  
  # Using outlier package
  if(length(clean_column) > 0) {  
    outlier_pkg <- outlier(clean_column)
  } else {
    outlier_pkg <- numeric(0) 
  }
  
  # Using IQR method
  outliers_iqr <- find_iqr_outliers(numerical_data[[var_name]])
  
  # Combine and make unique
  all_outliers <- unique(c(outlier_pkg, outliers_iqr))
  
  # Store results
  variable_names <- c(variable_names, var_name)
  outlier_counts <- c(outlier_counts, length(all_outliers))
}

# Create a data frame with results and sort in descending order of outlier counts
result <- data.frame(variable = variable_names, count = outlier_counts)
result <- result[order(-result$count), ]

print(result)


