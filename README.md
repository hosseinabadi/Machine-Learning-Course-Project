## Project: Purchase After Ad Click Prediction
Problem Statement:
In this problem, we want to predict whether a product will be bought after a consumber clicks on it's advertisement. Our input is information about features of the ad and the consumer and our output is a binary label which shows whether or not the product is bought after the cosumer has clicked on the ad.  
I implemented a machine learning pipeline to predict whether or not customers will eventually purchase a product after clicking on its ad. The project consisted of data cleaning, exploratory data analysis(EDA), visualization, feature engineering,  one and finally containerizing different parts with docker and made a pipeline of these containers so that the project could be deployed by users:  

### Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering:
Finding missing data, Drop redundant columns, Finding relation between missing values of different columns, Remove columns and rows with more than 90% missing values (because we cant to imputation), Imputation (Datawig), Removing outliers, one-hot-encoding, PCA.  


### Visualization:
Describing data, Distribution of column data (both numerical and categorical columns), Correlation of columns

### Models:
Testing and tuning different models such as Wide and Deep Learning, Light GBM, Multi-layer Perceptron Classifier to choose the most accurate.


## Codes
models : Cleaning code + Data visualization + Models  
main : sends request to Data_cleaner  
app + docker : Data_cleaner  
