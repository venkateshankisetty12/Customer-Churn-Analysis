# 📊 Customer Churn Analysis Using Machine Learning

## 📌 Project Overview

Customer churn is one of the major challenges faced by businesses, especially subscription-based companies.

**Customer Churn Analysis** is a machine learning project that analyzes customer information and predicts whether a customer is likely to **leave (churn)** the company.

The project combines **Exploratory Data Analysis (EDA), Data Preprocessing, Data Visualization, Feature Engineering, and Machine Learning** to identify the important factors associated with customer churn.

The main goal is to help businesses understand:

* Who is likely to churn?
* Why are customers leaving?
* Which factors are strongly associated with churn?
* How can businesses identify high-risk customers early?

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Analyze customer demographic and service-related information.
2. Perform data cleaning and preprocessing.
3. Explore customer behavior using EDA.
4. Identify important factors associated with customer churn.
5. Prepare the data for machine learning.
6. Build classification models to predict customer churn.
7. Evaluate model performance using appropriate metrics.
8. Identify customers who have a higher probability of churning.
9. Provide business insights that can support customer retention strategies.

---

## 🛠️ Technologies & Tools Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Development Environment

* Jupyter Notebook / Google Colab

### Machine Learning

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
Customer-Churn-Analysis/
│
├── 📁 data/
│   └── customer_churn.csv
│
├── 📁 notebooks/
│   └── customer_churn_analysis.ipynb
│
├── 📁 images/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_tenure.png
│   └── model_comparison.png
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 customer_churn_model.py
```

---

# 📊 Dataset

The dataset contains customer-level information that can be used to understand customer behavior and predict churn.

Typical features include:

| Feature          | Description                              |
| ---------------- | ---------------------------------------- |
| customerID       | Unique customer identifier               |
| gender           | Customer gender                          |
| SeniorCitizen    | Whether the customer is a senior citizen |
| Partner          | Whether the customer has a partner       |
| Dependents       | Whether the customer has dependents      |
| tenure           | Number of months the customer has stayed |
| PhoneService     | Whether phone service is active          |
| MultipleLines    | Multiple phone lines                     |
| InternetService  | Type of internet service                 |
| OnlineSecurity   | Online security subscription             |
| OnlineBackup     | Online backup subscription               |
| DeviceProtection | Device protection subscription           |
| TechSupport      | Technical support subscription           |
| StreamingTV      | Streaming TV subscription                |
| StreamingMovies  | Streaming movies subscription            |
| Contract         | Customer contract type                   |
| PaperlessBilling | Whether paperless billing is enabled     |
| PaymentMethod    | Customer payment method                  |
| MonthlyCharges   | Monthly amount paid by customer          |
| TotalCharges     | Total amount paid by customer            |
| Churn            | Whether the customer left the company    |

### Target Variable

The target variable is:

```text
Churn
```

Possible values:

```text
Yes → Customer churned
No  → Customer stayed
```

This makes the problem a **binary classification problem**.

---

# 🔍 Business Problem

A company may have thousands of customers, but it is difficult to manually identify which customers are at risk of leaving.

If the company can predict customers who are likely to churn, it can take preventive actions such as:

* Offering personalized discounts
* Improving customer support
* Providing better service plans
* Offering loyalty benefits
* Contacting high-risk customers
* Improving customer experience

Therefore, machine learning can be used to predict customer churn before it happens.

---

# 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Encoding & Scaling
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Churn Prediction
     ↓
Business Insights
```

---

# 1️⃣ Import Required Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

---

# 2️⃣ Load the Dataset

```python
df = pd.read_csv("customer_churn.csv")

df.head()
```

To understand the dataset:

```python
df.shape
```

```python
df.info()
```

```python
df.describe()
```

---

# 3️⃣ Data Cleaning

## Check Missing Values

```python
df.isnull().sum()
```

Missing values need to be handled before building the machine learning model.

For example:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Then check missing values again:

```python
df.isnull().sum()
```

Missing numerical values can be handled using median or another appropriate strategy.

Example:

```python
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
```

---

# 4️⃣ Remove Unnecessary Columns

Customer IDs generally do not provide useful predictive information.

Therefore:

```python
df.drop("customerID", axis=1, inplace=True)
```

---

# 5️⃣ Exploratory Data Analysis

EDA helps us understand patterns and relationships in the data.

## Churn Distribution

```python
sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")
plt.show()
```

This helps us understand how many customers stayed and how many customers churned.

---

# 👥 Churn by Contract Type

```python
sns.countplot(
    x="Contract",
    hue="Churn",
    data=df
)

plt.title("Churn by Contract Type")
plt.xticks(rotation=20)
plt.show()
```

This analysis helps determine whether contract type is associated with customer churn.

---

# 💰 Monthly Charges vs Churn

```python
sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df
)

plt.title("Monthly Charges vs Churn")
plt.show()
```

This helps analyze whether customers paying different monthly amounts show different churn patterns.

---

# ⏳ Tenure vs Churn

```python
sns.boxplot(
    x="Churn",
    y="tenure",
    data=df
)

plt.title("Tenure vs Churn")
plt.show()
```

Customer tenure can help identify whether newer customers behave differently from long-term customers.

---

# 💳 Payment Method vs Churn

```python
sns.countplot(
    x="PaymentMethod",
    hue="Churn",
    data=df
)

plt.title("Churn by Payment Method")
plt.xticks(rotation=45)
plt.show()
```

This helps identify whether churn patterns differ across payment methods.

---

# 6️⃣ Feature Engineering

Machine learning algorithms require numerical inputs.

Categorical columns therefore need to be converted into numerical form.

For example:

```python
df = pd.get_dummies(
    df,
    drop_first=True
)
```

The target variable can also be converted:

```python
df["Churn_Yes"] = df["Churn_Yes"].astype(int)
```

---

# 7️⃣ Define Features and Target

```python
X = df.drop("Churn_Yes", axis=1)

y = df["Churn_Yes"]
```

Where:

* `X` = Independent variables/features
* `y` = Target variable

The target represents whether the customer churned.

---

# 8️⃣ Train-Test Split

The dataset is divided into training and testing datasets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Why use Train-Test Split?

The training data is used to train the model, while the testing data is used to evaluate how well the model performs on unseen data.

---

# 9️⃣ Feature Scaling

Some algorithms perform better when numerical variables are on a similar scale.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

**Important:** The scaler is fitted only on the training data to avoid data leakage.

---

# 🤖 1. Logistic Regression

Logistic Regression is a common algorithm for binary classification problems.

```python
log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(
    X_train_scaled,
    y_train
)

y_pred_log = log_model.predict(
    X_test_scaled
)
```

---

# 🌳 2. Decision Tree

Decision Tree is a tree-based classification algorithm.

```python
dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

y_pred_dt = dt_model.predict(
    X_test
)
```

---

# 🌲 3. Random Forest

Random Forest combines multiple decision trees to improve predictive performance and reduce overfitting compared with a single tree.

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(
    X_test
)
```

---

# 📈 Model Evaluation

Several metrics can be used to evaluate classification models.

## Accuracy

```python
accuracy_score(y_test, y_pred_rf)
```

Accuracy represents the percentage of total predictions that were correct.

---

## Precision

```python
precision_score(y_test, y_pred_rf)
```

Precision answers:

> Of the customers predicted as churners, how many actually churned?

---

## Recall

```python
recall_score(y_test, y_pred_rf)
```

Recall answers:

> Of all the customers who actually churned, how many did the model correctly identify?

For churn prediction, recall can be particularly important because missing a customer who is likely to churn may mean losing an opportunity for retention.

---

## F1 Score

```python
f1_score(y_test, y_pred_rf)
```

F1-score provides a balance between precision and recall.

---

# 📊 Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)
```

This provides:

* Precision
* Recall
* F1-score
* Support

for each class.

---

# 🔲 Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    y_pred_rf
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

The confusion matrix contains:

* True Positive
* True Negative
* False Positive
* False Negative

---

# 🏆 Model Comparison

The models can be compared using several evaluation metrics.

Example:

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression | Add Result | Add Result | Add Result | Add Result |
| Decision Tree       | Add Result | Add Result | Add Result | Add Result |
| Random Forest       | Add Result | Add Result | Add Result | Add Result |

> **Note:** Replace the placeholder values with the actual results generated from your dataset. Do not add assumed model scores to the README.

---

# 🔎 Feature Importance

Random Forest can be used to identify features that contributed strongly to predictions.

```python
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.head(10)
```

Visualization:

```python
plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")
plt.show()
```

This helps understand which variables were influential in the model.

---

# 💡 Key Business Insights

The analysis can be used to identify patterns such as:

### 1. Contract Type

Customers on shorter-term contracts may exhibit different churn behavior compared with customers on longer-term contracts.

### 2. Customer Tenure

Customers with shorter tenure can represent an important segment for early retention efforts.

### 3. Monthly Charges

Customers with higher monthly charges may show different churn patterns and can be analyzed as a potential risk segment.

### 4. Services

Internet services, technical support, security services, and other subscribed services can be analyzed for their relationship with churn.

### 5. Payment Method

Different payment methods may show different customer behavior and churn patterns.

> These are analytical areas to investigate. The exact conclusions should be based on the results obtained from the dataset.

---

# 🎯 Business Recommendations

Based on the analysis, a company could consider:

* Identifying high-risk customers using the churn prediction model.
* Providing personalized retention offers.
* Improving onboarding for new customers.
* Reviewing pricing and service plans.
* Providing proactive technical support.
* Offering loyalty benefits to long-term customers.
* Monitoring customers with multiple churn-risk indicators.
* Using churn probability to prioritize customer-retention campaigns.

---

# 📌 Example Prediction

After training the model, we can predict whether a new customer is likely to churn.

```python
prediction = rf_model.predict(new_customer)

if prediction[0] == 1:
    print("Customer is likely to churn")
else:
    print("Customer is likely to stay")
```

The model can also provide a probability:

```python
probability = rf_model.predict_proba(new_customer)

print(probability)
```

This can be useful for ranking customers according to estimated churn risk.

---

# ⚠️ Important Considerations

## Class Imbalance

Customer churn datasets can contain more non-churned customers than churned customers.

Therefore, accuracy alone may not provide the complete picture.

Metrics such as:

* Precision
* Recall
* F1-score

should also be considered.

---

## Data Leakage

Information that would not be available at prediction time should not be used as a feature.

The preprocessing pipeline should be designed carefully to avoid using information from the test set during training.

---

## Model Interpretation

Machine learning predictions should be combined with business understanding.

Feature importance indicates variables associated with model predictions; it does **not automatically prove that those variables cause churn**.

---

# 🚀 Future Improvements

This project can be further improved by:

* Hyperparameter tuning
* Cross-validation
* Handling class imbalance using appropriate techniques
* Trying Gradient Boosting / XGBoost
* Creating a complete ML pipeline
* Saving the trained model using Joblib
* Building a Streamlit prediction application
* Creating a Power BI dashboard for business users
* Deploying the model as a web application
* Monitoring model performance after deployment

---

# 📊 Possible Power BI Dashboard

The machine learning analysis can also be combined with Power BI to create a business dashboard.

### KPI Cards

* Total Customers
* Churned Customers
* Churn Rate
* Average Monthly Charges
* Average Tenure

### Visualizations

* Churn by Contract
* Churn by Internet Service
* Churn by Payment Method
* Churn by Tenure
* Churn by Monthly Charges
* Churn by Customer Demographics

### Filters

* Gender
* Contract
* Internet Service
* Payment Method
* Tenure
* Senior Citizen

This combination demonstrates both **Data Analytics and Machine Learning skills**.

---

# 🧠 Skills Demonstrated

This project demonstrates practical knowledge of:

### Python

* Pandas
* NumPy
* Data manipulation
* Data preprocessing

### Data Analysis

* Exploratory Data Analysis
* Missing-value handling
* Categorical data analysis
* Statistical summaries

### Data Visualization

* Matplotlib
* Seaborn
* Distribution analysis
* Relationship analysis

### Machine Learning

* Classification
* Logistic Regression
* Decision Trees
* Random Forest
* Feature engineering
* Train-test split
* Feature scaling
* Model evaluation
* Feature importance

### Business Analytics

* Customer behavior analysis
* Churn-risk identification
* Customer retention insights
* Data-driven decision making

---

# 📁 Requirements

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or create a `requirements.txt` file:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Then install:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project directory

```bash
cd Customer-Churn-Analysis
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Open the notebook

```bash
jupyter notebook
```

### Step 5: Open

```text
customer_churn_analysis.ipynb
```

### Step 6: Run all cells

Execute the notebook from beginning to end to reproduce the analysis and model results.

---

# 📷 Project Screenshots

Add screenshots of your important analysis and visualizations here.

Example:

```text
images/
├── churn_distribution.png
├── churn_by_contract.png
├── churn_by_tenure.png
├── correlation_heatmap.png
├── feature_importance.png
└── model_comparison.png
```

You can display them in GitHub using:

```markdown
![Churn Distribution](images/churn_distribution.png)
```

---

# 📌 Project Outcome

The project demonstrates how customer data can be transformed into actionable insights using:

**Data Cleaning → EDA → Feature Engineering → Machine Learning → Model Evaluation → Business Insights**

The final model can help identify customers who may be at higher risk of churn, allowing businesses to investigate those customers and consider appropriate retention strategies.

---

# 👨‍💻 Author

**Ankisetty Venkateswarlu**

Aspiring Data Analyst | Python | SQL | Excel | Power BI | Machine Learning

### Connect with me

* LinkedIn: Add your LinkedIn profile
* GitHub: Add your GitHub profile

---

# ⭐ If you found this project useful

If you find this project useful for learning Data Analytics or Machine Learning, consider giving the repository a ⭐.

---

## 📌 Interview Explanation — 60 Seconds

> "I worked on a Customer Churn Analysis project using Python and Machine Learning. The main objective was to analyze customer behavior and predict whether a customer was likely to churn. I started by cleaning the dataset, handling missing values and converting categorical variables into numerical features. Then I performed exploratory data analysis using Pandas, Matplotlib and Seaborn to understand factors such as contract type, tenure, monthly charges and payment methods. After preprocessing the data, I trained classification models including Logistic Regression, Decision Tree and Random Forest. I evaluated the models using accuracy, precision, recall and F1-score, and also analyzed the confusion matrix and feature importance. Finally, I converted the analytical findings into business insights that could help a company identify high-risk customers and improve customer retention."

---

## 📌 Resume Project Description

**Customer Churn Analysis | Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn**

* Analyzed customer data to identify patterns and factors associated with customer churn using Python and exploratory data analysis.
* Cleaned and preprocessed customer data, handled missing values, encoded categorical variables, and prepared features for machine learning.
* Built and evaluated Logistic Regression, Decision Tree, and Random Forest classification models using accuracy, precision, recall, F1-score, and confusion matrix.
* Analyzed feature importance and translated model findings into actionable customer-retention insights.
