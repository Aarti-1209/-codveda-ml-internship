"""
Codveda Technologies - Machine Learning Internship
Level 2 (Intermediate) - Task 2: Decision Trees for Classification

Description: Build a decision tree classifier to predict the species of
flowers using the Iris dataset.

Objectives:
1. Train a decision tree on a labeled dataset (Iris dataset)
2. Visualize the tree structure
3. Prune the tree to prevent overfitting
4. Evaluate the model using classification metrics (accuracy, F1-score)

Tools: Python, scikit-learn, pandas, matplotlib
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD THE DATASET
# ---------------------------------------------------------
df = pd.read_csv("../data/iris.csv")
print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df['species'].value_counts())

X = df.drop(columns=['species'])
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. TRAIN AN INITIAL (UNPRUNED) DECISION TREE
# ---------------------------------------------------------
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

y_pred_full = full_tree.predict(X_test)
print(f"\n--- Full (unpruned) tree ---")
print(f"Depth: {full_tree.get_depth()}, Leaves: {full_tree.get_n_leaves()}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_full):.3f}")
print(f"Test F1-score (macro): {f1_score(y_test, y_pred_full, average='macro'):.3f}")

# ---------------------------------------------------------
# 3. PRUNE THE TREE TO PREVENT OVERFITTING
# ---------------------------------------------------------
# Use GridSearchCV to find the best max_depth / min_samples_leaf combination
param_grid = {
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_leaf': [1, 2, 4, 6]
}
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

best_tree = grid_search.best_estimator_
print(f"\n--- Pruned (best) tree ---")
print("Best hyperparameters:", grid_search.best_params_)
print(f"Depth: {best_tree.get_depth()}, Leaves: {best_tree.get_n_leaves()}")

y_pred_pruned = best_tree.predict(X_test)
acc_pruned = accuracy_score(y_test, y_pred_pruned)
f1_pruned = f1_score(y_test, y_pred_pruned, average='macro')

print(f"Test Accuracy: {acc_pruned:.3f}")
print(f"Test F1-score (macro): {f1_pruned:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_pruned))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_pruned))

# ---------------------------------------------------------
# 4. VISUALIZE THE PRUNED TREE STRUCTURE
# ---------------------------------------------------------
plt.figure(figsize=(16, 9))
plot_tree(
    best_tree,
    feature_names=X.columns,
    class_names=best_tree.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Pruned Decision Tree - Iris Species Classification")
plt.tight_layout()
plt.savefig("decision_tree_structure.png", dpi=150)
print("\nTree visualization saved as decision_tree_structure.png")
