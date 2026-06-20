"""
Codveda Technologies - Machine Learning Internship
Level 2 (Intermediate) - Task 1: Logistic Regression for Binary Classification

Description: Implement a logistic regression model to predict whether a
customer will churn, using the telecom customer churn dataset.

Objectives:
1. Load and preprocess the dataset
2. Train a logistic regression model using scikit-learn
3. Interpret model coefficients and the odds ratio
4. Evaluate the model using accuracy, precision, recall, and the ROC curve

Tools: Python, pandas, scikit-learn, matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS THE DATASET
# ---------------------------------------------------------
# Dataset already comes pre-split (80% train / 20% test)
train_df = pd.read_csv("../data/churn-bigml-80.csv")
test_df = pd.read_csv("../data/churn-bigml-20.csv")

print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
print("\nChurn distribution (train):\n", train_df['Churn'].value_counts())

# Encode binary categorical columns
binary_cols = ['International plan', 'Voice mail plan']
for col in binary_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# One-hot encode 'State' (many categories, no ordinal meaning)
train_df = pd.get_dummies(train_df, columns=['State'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['State'], drop_first=True)

# Align columns between train/test (in case some states only appear in one set)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# Target variable: Churn (convert boolean to int)
train_df['Churn'] = train_df['Churn'].astype(int)
test_df['Churn'] = test_df['Churn'].astype(int)

X_train = train_df.drop(columns=['Churn'])
y_train = train_df['Churn']
X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn']

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. TRAIN THE LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 3. INTERPRET COEFFICIENTS AND ODDS RATIOS
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0],
    'Odds_Ratio': np.exp(model.coef_[0])
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 features by impact on churn:\n", coef_df.head(10))

print("""
Interpretation:
- Odds Ratio > 1 means the feature INCREASES the odds of churn.
- Odds Ratio < 1 means the feature DECREASES the odds of churn.
- E.g., 'Customer service calls' and 'International plan' typically increase
  churn odds, while 'Voice mail plan' tends to reduce it.
""")

# ---------------------------------------------------------
# 4. EVALUATE THE MODEL
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"AUC:       {auc:.3f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 5. ROC CURVE
# ---------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Customer Churn)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("logistic_regression_roc_curve.png", dpi=150)
print("\nROC curve saved as logistic_regression_roc_curve.png")
