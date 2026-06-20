"""
Codveda Technologies - Machine Learning Internship
Level 1 (Basic) - Task 1: Data Preprocessing for Machine Learning

Description: Preprocess a raw stock prices dataset to make it ready for machine learning.

Objectives:
1. Handle missing data (filling with mean/median, dropping)
2. Encode categorical variables (one-hot encoding / label encoding)
3. Normalize or standardize numerical features
4. Split the dataset into training and testing sets

Tools: Python, pandas, scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. LOAD THE RAW DATASET
# ---------------------------------------------------------
df = pd.read_csv("../data/Stock_Prices_Data_Set.csv")
print("Original dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)

# For speed/manageability, work with a sample of stocks (top 20 by frequency)
top_symbols = df['symbol'].value_counts().head(20).index
df = df[df['symbol'].isin(top_symbols)].reset_index(drop=True)
print("\nSampled dataset shape (top 20 symbols):", df.shape)

# The sampled slice happens to have no missing values, so inject a small,
# realistic amount of missingness (2%) into numeric columns to genuinely
# demonstrate the missing-data handling step on this dataset.
np.random.seed(42)
for col in ['open', 'high', 'low']:
    missing_idx = df.sample(frac=0.02, random_state=42).index
    df.loc[missing_idx, col] = np.nan

# ---------------------------------------------------------
# 2. HANDLE MISSING DATA
# ---------------------------------------------------------
print("\nMissing values BEFORE handling:\n", df.isnull().sum())

# Fill missing numeric values with the column median (robust to outliers)
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print("\nMissing values AFTER handling:\n", df.isnull().sum())

# ---------------------------------------------------------
# 3. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------
# 'symbol' is categorical -> use Label Encoding (many unique categories)
le = LabelEncoder()
df['symbol_encoded'] = le.fit_transform(df['symbol'])

# Extract useful date features instead of keeping raw date string
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

print("\nSample after encoding:\n", df[['symbol', 'symbol_encoded', 'year', 'month', 'day']].head())

# ---------------------------------------------------------
# 4. NORMALIZE / STANDARDIZE NUMERICAL FEATURES
# ---------------------------------------------------------
features_to_scale = ['open', 'high', 'low', 'volume']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("\nScaled features summary:\n", df[features_to_scale].describe())

# ---------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------
feature_cols = ['symbol_encoded', 'open', 'high', 'low', 'volume', 'year', 'month', 'day']
target_col = 'close'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Save the processed datasets for reuse
df.to_csv("stock_prices_preprocessed.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPreprocessing complete. Files saved:")
print(" - stock_prices_preprocessed.csv")
print(" - X_train.csv, X_test.csv, y_train.csv, y_test.csv")
