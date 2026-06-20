"""
Codveda Technologies - Machine Learning Internship
Level 1 (Basic) - Task 2: Build a Simple Linear Regression Model

Description: Build a linear regression model to predict a continuous variable
(house prices) using the House Prices dataset.

Objectives:
1. Load a dataset and preprocess it
2. Train a linear regression model using scikit-learn
3. Interpret the model coefficients
4. Evaluate the model using R-squared and Mean Squared Error (MSE)

Tools: Python, pandas, scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS THE DATASET
# ---------------------------------------------------------
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df = pd.read_csv("../data/house_Prediction_Data_Set.csv", header=None, names=column_names, sep=r"\s+")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum().sum())
print("\nSummary statistics:\n", df.describe())

# Target variable: MEDV (Median value of owner-occupied homes, in $1000s)
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Scale features (helps with coefficient interpretability/comparability)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---------------------------------------------------------
# 2. TRAIN/TEST SPLIT AND MODEL TRAINING
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. INTERPRET MODEL COEFFICIENTS
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nModel Intercept:", model.intercept_)
print("\nFeature Coefficients (sorted by absolute impact):\n", coef_df)

print("""
Interpretation:
- Each coefficient represents the change in predicted house price (in $1000s)
  for a one standard-deviation increase in that feature, holding others constant.
- RM (average rooms per dwelling) typically has a strong POSITIVE effect on price.
- LSTAT (% lower status population) typically has a strong NEGATIVE effect on price.
""")

# ---------------------------------------------------------
# 4. EVALUATE THE MODEL
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2): {r2:.3f}")

# ---------------------------------------------------------
# 5. VISUALIZE: ACTUAL VS PREDICTED
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual House Price ($1000s)")
plt.ylabel("Predicted House Price ($1000s)")
plt.title("Linear Regression: Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("linear_regression_actual_vs_predicted.png", dpi=150)
print("\nPlot saved as linear_regression_actual_vs_predicted.png")
