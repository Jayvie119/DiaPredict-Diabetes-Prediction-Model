# DiaPredict: Diabetes Prediction Model

This project is a machine learning model that predicts whether a patient is likely to have diabetes or not, based on diagnostic measurements. The data used is from the popular **Pima Indians Diabetes Database**.

# Dataset: 
> https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## 📁 Project Structure
- diabetes.csv # Dataset
- model_training.py # Python script for cleaning, training, evaluating the model
- diabetes_model.pkl # Saved model file
- requirements.txt # Required dependencies
- README.md # Project documentation

## Description

This repository currently features a **Random Forest Classifier** for predicting diabetes. The data undergoes cleaning where zero values are treated as missing and imputed using the median of respective columns.

> **Note:** I will be testing multiple prediction models over time (e.g., Logistic Regression, SVM, KNN, XGBoost) and updating the results accordingly.

## Data Preprocessing

- Missing/invalid values (e.g. 0 in `Glucose`, `Insulin`, etc.) are replaced with `NaN`.
- Missing values are filled using the **median** of each column.
- Features are split into `X` (independent) and `y` (dependent / Outcome).
- Data is split into training and testing sets (80-20 split).

## Model Training & Evaluation

- **Model:** RandomForestClassifier from `sklearn.ensemble`
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Feature Importance** is also visualized.

Example output:
```
Accuracy:  0.7467532467532467
Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.80      0.80        99
           1       0.64      0.65      0.65        55

    accuracy                           0.75       154
   macro avg       0.72      0.73      0.73       154
weighted avg       0.75      0.75      0.75       154

Confusion Matrix:
 [[79 20]
 [19 36]]
```

## Feature Importance Chart

The most influential features in predicting diabetes are visualized using a horizontal bar chart.
![image](https://github.com/user-attachments/assets/08f9f6d5-3103-44f1-a041-b722e7a94c88)

## Save & Load Model

To save the trained model:
```python
import joblib
joblib.dump(model, 'diabetes_model.pkl')
```
To load it again:
```python
loaded_model = joblib.load('diabetes_model.pkl')
```

## Setup
1. Clone the repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the model training
   ```
   python Diabetes_Prediction_Model.py
   ```
   



   
