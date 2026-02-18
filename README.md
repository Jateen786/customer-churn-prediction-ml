# customer-churn-prediction-ml
## Overview
This project predicts whether a telecom customer will churn (leave the service) using machine learning classification models.

The goal is to help businesses identify customers at risk and take preventive retention actions.

## Dataset
IBM Telco Customer Churn Dataset  
Contains ~7,000+ customer records with demographics, account information, and service usage.

Target Variable:
Churn (1 = customer leaves, 0 = customer stays)

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Steps Performed
1. Data Cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Label Encoding categorical features
4. Train-Test Split
5. Model Training (Logistic Regression & Random Forest)
6. Model Evaluation using:
   - Accuracy
   - Precision & Recall
   - Confusion Matrix
   - ROC-AUC Score

## Results
Random Forest performed better than Logistic Regression with higher ROC-AUC score and improved classification performance.

## Business Impact
The model helps identify high-risk customers, allowing telecom companies to:
- Offer targeted discounts
- Improve customer retention
- Reduce revenue loss

## How to Run
1. Clone repository
2. Install requirements:
   pip install pandas numpy scikit-learn matplotlib seaborn
3. Run churn.ipynb
