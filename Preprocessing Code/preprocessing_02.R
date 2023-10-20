# NOTE: This is starting on splitting the data. 

# Load packages 
library(tidyverse)
library(skimr)
library(lubridate)
library(timetk)
library(forecast)

# Load data 
data <- read_csv('data/raw_data/owid-covid-data.csv')
source('preprocessing_01.R')

######################################################################################
# Q: Would we be using `data_clean5` or `data_clean6`?

# Ensure the data is in chronological order
data_sorted <- data_clean5 %>% arrange(date)

# Split the data into training and testing sets

# ---- Option 1
training_data <- data_sorted %>% filter(date <= as.Date("INSERT DATE"))
testing_data <- data_sorted %>% filter(date > as.Date("INSERT DATE"))

# ---- Option 2: timetk package
splits <- tk_time_series_cv(data_sorted, 
                            assess = "SIZE OF TEST SET", 
                            skip = "GAP BETWEEN SLICES", 
                            slice_limit = MAX NUM OF ROLLING WINDOWS)

# ---- Option 2: forecast package
training_data <- head(data_clean5, round(length(data_clean5) * 0.7)) # only showing up 42 obs??
h <- length(data_clean5) - length(training_data) # 0L??
testing_data <- tail(data_clean5, h)
autoplot(training_data) + autolayer(testing_data)
