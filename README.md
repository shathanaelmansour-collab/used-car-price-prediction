# 🚗 Used Car Price Prediction \| Machine Learning Regression

## 📌 Project Overview

This project is an end-to-end **Machine Learning Regression** project
designed to predict the selling price of a used car based on its
characteristics.

The project covers data cleaning, exploratory data analysis (EDA),
feature engineering, preprocessing, model comparison, hyperparameter
tuning, model evaluation, SHAP explainability, Scikit-learn pipeline
development, and Streamlit deployment.

## 🎯 Project Objective

The objective is to estimate used-car selling prices from features
including car age, kilometers driven, fuel type, seller type,
transmission, ownership history, mileage, engine capacity, maximum
power, and number of seats.

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   SHAP
-   Joblib
-   Streamlit

## 🧹 Data Preprocessing

Missing numerical values were handled using imputation. Duplicate
observations were removed. Numerical values were extracted from
text-based mileage, engine, and maximum-power fields. Vehicle year
information was transformed into `car_age`.

Categorical variables such as fuel type, seller type, transmission, and
owner are handled in the deployment-ready Scikit-learn pipeline using
`OneHotEncoder`.

## 🤖 Models Evaluated

### Linear Regression

-   **R²:** \~0.60
-   **MAE:** \~170,706
-   **RMSE:** \~297,063

### Random Forest Regressor

-   **R²:** \~0.91
-   **MAE:** \~76,615
-   **RMSE:** \~138,708

### Gradient Boosting Regressor

-   **R²:** \~0.88
-   **MAE:** \~91,211
-   **RMSE:** \~160,715

Random Forest provided the strongest performance among the tested
models.

## ⚙️ Hyperparameter Tuning

`RandomizedSearchCV` with cross-validation was used to optimize the
Random Forest model. Parameters explored included `n_estimators`,
`max_depth`, `min_samples_split`, `min_samples_leaf`, and
`max_features`.

After tuning, the Random Forest achieved approximately:

-   **R²:** 0.918
-   **MAE:** 75,329
-   **RMSE:** 134,488

## 📊 Model Evaluation

The model was evaluated using R², MAE, MSE, and RMSE.
Actual-vs-predicted and residual plots were also used to inspect model
performance.

## 🔍 Feature Importance

Important features included:

1.  Maximum Power
2.  Car Age
3.  Kilometers Driven
4.  Mileage
5.  Engine Capacity

## 🧠 SHAP Explainability

SHAP was used to understand the model globally and explain individual
predictions. Higher maximum power generally pushed model predictions
upward, while greater car age generally pushed predictions downward.

SHAP values explain the behavior of the trained model and should not be
interpreted as proof of causal relationships.

## 🔄 Machine Learning Pipeline

The deployment-ready workflow combines preprocessing and prediction into
a Scikit-learn Pipeline:

`Raw Car Data → Missing Value Handling → One-Hot Encoding → Random Forest → Price Prediction`

## 🌐 Streamlit Application

The Streamlit application accepts car age, kilometers driven, mileage,
engine capacity, maximum power, seats, fuel type, seller type,
transmission, and owner type, then returns an estimated selling price.

## 📂 Project Structure

``` text
CarPriceProject/
├── app.py
├── car_price_pipeline.pkl
├── requirements.txt
├── README.md
└── CarPriceRegressionModel.ipynb
```

## ▶️ Run the Project Locally

``` bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Requirements

``` text
streamlit
pandas
numpy
scikit-learn
joblib
```

Use a Scikit-learn version compatible with the version used to create
the saved pipeline.

## 💡 Key Learning Outcomes

-   End-to-end regression workflow
-   Data cleaning and preprocessing
-   Feature engineering
-   Model comparison
-   Regression evaluation metrics
-   Hyperparameter optimization
-   Random Forest regression
-   Feature importance
-   SHAP model explainability
-   Scikit-learn pipelines
-   Model serialization with Joblib
-   Streamlit application development and deployment

## 👩‍💻 Author

**Shatha Mansour**

Data Science & Artificial Intelligence Graduate

Interested in Data Science, Machine Learning, Artificial Intelligence,
Data Analytics, and Business Intelligence.
