### Credit Card Approval System
![Python](https://img.shields.io/badge/Python-3.12.4-blueviolet)
![Tensorflow](https://img.shields.io/badge/ML-Tensorflow-fcba03)
![Colab](https://img.shields.io/badge/Editor-GColab-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-brightgreen)
![Heroku](https://img.shields.io/badge/Backend-Heroku-pink)

![intro](card.png)

## Overview
The **Credit Card Approval System** is a sophisticated machine learning-driven application designed to predict the approval status of credit card applications. Leveraging advanced algorithms, this system evaluates various applicant attributes to provide reliable and efficient predictions, streamlining the approval process while mitigating financial risks.

## Key Features
- **Data Preprocessing**: Robust handling of missing data, categorical variable encoding, and feature scaling to ensure data quality.
- **Model Training**: Implementation of powerful algorithms like Decision Trees, Logistic Regression, and Random Forest for model training.
- **Model Evaluation**: In-depth performance analysis using metrics such as accuracy, precision, recall, and F1-score.
- **Deployment**: Seamless deployment using frameworks like Flask or Streamlit, with backend support on Heroku.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aryanc381/Credit-Card-Approval-System.git
   ```

2. **Install Required Packages**:
   ```bash
   pip install scikit-learn numpy pandas matplotlib seaborn
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Model Backend**:
   ```bash
   card_model.pkl
   ```

5. **Modify the Model**:
   ```bash
   python main.ipynb
   ```

## Dataset
The dataset utilized is sourced from [Kaggle](https://www.kaggle.com/datasets/rohit265/credit-card-eligibility-data-determining-factors) and includes various features concerning applicants' financial history, personal information, and creditworthiness.

## Model Performance
- **Accuracy**: 96%
- **Precision**: 52%
- **Recall**: 99%
- **F1-Score**: 98%

## Model Implementation Details

### Data Preprocessing
Data preprocessing is a crucial step to ensure the quality and reliability of the model. Here's an overview of the steps involved:

1. **Importing Libraries**:
   ```python
   import numpy as np
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   ```

2. **Feature Engineering**:
   Understanding and defining each feature is vital for the model's accuracy. For instance:
   - `CODE_GENDER`: Indicates the applicant's gender.
   - `FLAG_OWN_CAR`: Denotes whether the applicant owns a car.
   - `AMT_INCOME_TOTAL`: Represents the applicant's annual income.

3. **Handling Missing Values**:
   ```python
   data = pd.read_csv('/content/train.csv')
   data.isnull().sum()
   ```

   This command helps identify missing values, allowing us to decide whether to impute or drop them.

4. **Label Encoding**:
   Categorical variables like `CODE_GENDER` are converted into numerical values using Label Encoding.

### Model Implementation
The core of the Credit Card Approval System lies in the model implementation:

1. **Train-Test Split**:
   Splitting the data into training and testing sets ensures the model is well-evaluated.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

2. **Decision Tree Classifier**:
   A Decision Tree model is employed to classify whether a claimant is eligible for a credit card.
   ```python
   from sklearn.tree import DecisionTreeClassifier
   model = DecisionTreeClassifier()
   model.fit(X_train, y_train)
   ```

3. **Model Evaluation**:
   Evaluate the model's performance using various metrics:
   ```python
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
   y_pred = model.predict(X_test)
   print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
   print(f"Precision: {precision_score(y_test, y_pred)}")
   print(f"Recall: {recall_score(y_test, y_pred)}")
   print(f"F1-Score: {f1_score(y_test, y_pred)}")
   ```

4. **Saving the Model**:
   To deploy the model, it needs to be saved after training:
   ```python
   import joblib
   joblib.dump(model, 'card_model.pkl')
   ```

## Brief Code Snippet: Adding a Decision Tree to a Dataset
A Decision Tree is used for both classification and regression tasks. Here's a concise demonstration of how to apply a Decision Tree to a dataset:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Loading and preprocessing data
data = pd.read_csv('dataset.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing and training the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

This snippet showcases how to train a Decision Tree model, make predictions, and evaluate its performance on a dataset. Decision Trees are particularly effective for classification tasks due to their ability to model complex decision boundaries.

## Contributions
We welcome contributions from the community! If you're interested in improving the project, feel free to fork the repository and submit a pull request.

---

For any queries related to this project, please reach out via the email provided in my profile section.
