"""
Codveda Technologies - Machine Learning Internship
Level 3 (Advanced) - Task 1: Build a Random Forest Classifier

Description: Implement a Random Forest model for classification on the
telecom customer churn dataset (a complex, real-world dataset).

Objectives:
1. Train a Random Forest model and tune hyperparameters (number of trees, max depth)
2. Evaluate the model using cross-validation and classification metrics
   (precision, recall, F1-score)
3. Perform feature importance analysis

Tools: Python, scikit-learn, pandas, matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS THE DATASET
# ---------------------------------------------------------
train_df = pd.read_csv("../data/churn-bigml-80.csv")
test_df = pd.read_csv("../data/churn-bigml-20.csv")

binary_cols = ['International plan', 'Voice mail plan']
for col in binary_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

train_df = pd.get_dummies(train_df, columns=['State'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['State'], drop_first=True)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

train_df['Churn'] = train_df['Churn'].astype(int)
test_df['Churn'] = test_df['Churn'].astype(int)

X_train = train_df.drop(columns=['Churn'])
y_train = train_df['Churn']
X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn']

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# ---------------------------------------------------------
# 2. HYPERPARAMETER TUNING (number of trees, max depth)
# ---------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("\nBest hyperparameters:", grid_search.best_params_)

# ---------------------------------------------------------
# 3. EVALUATE WITH CROSS-VALIDATION AND TEST SET METRICS
# ---------------------------------------------------------
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1')
print(f"\n5-Fold Cross-Validation F1 scores: {cv_scores}")
print(f"Mean CV F1-score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Test Set Performance ---")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 4. FEATURE IMPORTANCE ANALYSIS
# ---------------------------------------------------------
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 most important features:\n", importances.head(10))

top_n = 10
plt.figure(figsize=(9, 6))
top_features = importances.head(top_n).iloc[::-1]
plt.barh(top_features['Feature'], top_features['Importance'], color='seagreen')
plt.xlabel("Feature Importance")
plt.title(f"Random Forest - Top {top_n} Important Features (Churn Prediction)")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=150)
print("\nFeature importance chart saved as random_forest_feature_importance.png")
