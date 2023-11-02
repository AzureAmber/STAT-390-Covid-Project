#1. data spilt first (decide a date) - Training: Before 2023, Testing: After 2023

#2. imputation: for different models

# cindy - do linear model: remove all observations before certain date;
# willie do tree-based model: give very large numbers to those observations; 
# erica do neural network: 

#3. clustering using life_expectancy for:
#vaccinations, extreme_poverty, icu_units (and more) all these predictor variables

#4. only impute training set, impute testing set only if there's new feature added