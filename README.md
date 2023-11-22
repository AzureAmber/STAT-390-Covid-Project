# STAT-390-Covid-Project

Hello! This is the STAT 390 Github repository for our data science capstone project using a COVID-19 dataset. Our group consists of **Cindy Ha, Willie Xie, and Erica Zhang**. 

**Purpose:** 
The purpose of this project was to find a time series COVID-19 dataset and apply the skills we've developed in the R sequence, as well as new time-series methodologies, to investigate a research question. As the dataset we acquired from Kaggle spanned multiple countries and was updated regularly, we decided upon the following research task: To predict new cases of COVID-19 for G20 and G24 countries on the COUNTRY level. 

**Methodologies:**
Below is a quick outline of the methods we've applied throughout this project. 
- Checking stationarity with the ADF test
- K-Means clustering to impute missingness
- Using Cook's distance to determine outliers
- Building regressive models (ARIMA, Auto ARIMA, Prophet Univariate & Multivariate, XGBoost, Keras LSTM) with RMSE as performance metric
-   Tuning hyperparameters for these models with rolling origin validation sets and regular grids

**Instructions to Run Code:**
We have tried our best to keep this repository neat and intuitive.
- `data`: This folder contains our raw data, processed data (after intial data cleaning), and finalized data (training and testing for linear, tree-based, & neural network). 
- `Preprocessing Code`: This folder our EDA and data preprocessing steps. 
- `Reports`: This folder contains our weekly reports, proposal, and midterm report.
- `Models`: This folder contains all the R scripts for each member's models. Each R script will here will also specify the member's intitals and model type (e.g. `ch_xgboost.R`). Moreover, for Erica, the autoplots for each model's tuning result is saved in the `results` sub-directory under Models
- `Results`: This folder contains all the visualizations for each model, again categorized by member. This may include hyperparameter plots or Actual vs. Predicted New Cases per Country plots (e.g. `Results/willie/arima_argentina.png`). 

To run any of the R scripts or code, please make sure that the dataset has been loaded into the RStudio environment. We have also made sure to include these lines of code at the top. Additionally, some code may take more time to run (e.g. tuning), so we recommend looking at the commented out results or running the R script as a background job. 


**Dataset Source:** 
Our World in Data. (2023). Our World in Data - COVID-19 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/6559049

Best, 
*Group 2: Cindy Ha, Willie Xie, Erica Zhang*
