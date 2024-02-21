# Loan Approval Prediction Project

## Overview
This project aims to build machine learning models to predict whether a loan application should be approved or rejected based on various features provided by applicants. The models are trained using historical loan data, and their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Project Structure
The project is organized as follows:

Dataset/: Directory containing the dataset used for training and testing the models.
trained_models/: Directory containing the trained machine learning models saved as joblib files.
catboost_info/: Directory created by CatBoost during installation, containing configuration files and cached data.
README.md: This file providing an overview of the project.
loan_approval_prediction.ipynb: Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and testing.
test_data.csv: CSV file containing the testing data (X_test and y_test) used for evaluating the trained models.

## Dataset
The dataset used in this project contains information about loan applicants, including features such as number of dependents, education level, income, loan amount, credit score, and assets value. The target variable is the loan approval status (Approved/Rejected).

## Model Building
Four different machine learning algorithms are explored for building the loan approval prediction models:

1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. Gradient Boosting Classifier
4. CatBoost Classifier
   
Each model is trained using the training data and evaluated using various evaluation metrics to assess its performance.

## Usage
To use this project, follow these steps:

Clone the repository to your local machine:
-> git clone https://github.com/aditipadhiar/loan-prediction-ml.git

Install the required Python libraries:
-> pip install -r requirements.txt

Run the Jupyter Notebook loan_approval_prediction.ipynb to preprocess the data, train the models, and evaluate their performance.

Optionally, test the trained models using the provided testing dataset (test_data.csv).

## Results
The performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1-score. The results are summarized in the Jupyter Notebook and can be used to compare the effectiveness of different algorithms for loan approval prediction.

## Future Work
Possible future enhancements to this project include:

Fine-tuning hyperparameters of the models for better performance.
Exploring additional machine learning algorithms for model building.
Conducting further analysis to understand the factors influencing loan approval decisions.
