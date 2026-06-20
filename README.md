# 🤖 Machine Learning Internship — Codveda Technologies

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> One-month remote Machine Learning internship at **Codveda Technologies** (June–July 2026),
> covering supervised learning, classification, regression, and deep learning across
> real-world datasets.

---

## 📌 About This Repository

This repository contains all completed tasks for the Codveda Technologies Machine Learning
Internship. As per the internship guidelines, **2 out of 3 tasks were completed per level**
across 3 difficulty levels — Basic, Intermediate, and Advanced.

---

## 🗂️ Repository Structure

```
codveda-ml-internship/
│
├── data/                             # Datasets used across all tasks
│   ├── Stock_Prices_Data_Set.csv
│   ├── house_Prediction_Data_Set.csv
│   ├── churn-bigml-80.csv            # Training split (80%)
│   ├── churn-bigml-20.csv            # Test split (20%)
│   └── iris.csv
│
├── level1/                           # Basic Level
│   ├── task1_data_preprocessing.py
│   └── task2_linear_regression.py
│
├── level2/                           # Intermediate Level
│   ├── task1_logistic_regression.py
│   └── task2_decision_tree.py
│
├── level3/                           # Advanced Level
│   ├── task1_random_forest.py
│   └── task3_neural_network.py
│
└── README.md
```

---

## ✅ Tasks Completed

### 🟢 Level 1 — Basic

#### Task 1: Data Preprocessing for Machine Learning
- **Dataset**: Stock Prices Dataset (497K rows, 505 unique stock symbols)
- **What was done**:
  - Handled missing values using median imputation
  - Applied Label Encoding on the categorical `symbol` column
  - Standardized numerical features (open, high, low, volume) using `StandardScaler`
  - Extracted date features (year, month, day) from raw date strings
  - Split data into 80% training and 20% test sets
- **Tools**: Python, pandas, scikit-learn

#### Task 2: Build a Simple Linear Regression Model
- **Dataset**: House Prices Dataset (Boston Housing, 506 rows, 13 features)
- **What was done**:
  - Loaded and preprocessed the dataset (scaling, train/test split)
  - Trained a Linear Regression model to predict median house values (MEDV)
  - Interpreted model coefficients — identified `RM` (rooms) as top positive predictor
    and `LSTAT` (lower-status population %) as top negative predictor
  - Evaluated model performance using R² and MSE
- **Results**: R² = 0.669 | RMSE = 4.93
- **Tools**: Python, pandas, scikit-learn, matplotlib

---

### 🔵 Level 2 — Intermediate

#### Task 1: Logistic Regression for Binary Classification
- **Dataset**: Telecom Customer Churn (2,666 train / 667 test, pre-split)
- **What was done**:
  - Preprocessed dataset — encoded binary and categorical features, one-hot encoded `State`
  - Trained a Logistic Regression model with class balancing to handle imbalanced data
  - Interpreted odds ratios — `Customer service calls` and `International plan`
    increased churn risk; `Voice mail plan` reduced it
  - Evaluated using accuracy, precision, recall, F1-score, and ROC curve
- **Results**: Accuracy = 77.1% | AUC = 0.814
- **Tools**: Python, pandas, scikit-learn, matplotlib

#### Task 2: Decision Trees for Classification
- **Dataset**: Iris Flower Dataset (150 rows, 3 balanced classes)
- **What was done**:
  - Trained an initial unpruned Decision Tree (depth = 5)
  - Applied hyperparameter tuning via `GridSearchCV` to find optimal `max_depth`
    and `min_samples_leaf` — pruned tree to depth = 4
  - Visualized full tree structure with color-coded nodes
  - Evaluated using accuracy and macro F1-score
- **Results**: Accuracy = 93.3% | F1-score = 0.933
- **Tools**: Python, scikit-learn, pandas, matplotlib

---

### 🔴 Level 3 — Advanced

#### Task 1: Build a Random Forest Classifier
- **Dataset**: Telecom Customer Churn (same as Level 2 Task 1)
- **What was done**:
  - Tuned hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`)
    using `GridSearchCV` — best config: 200 trees, no max depth restriction
  - Evaluated using 5-fold cross-validation (Mean CV F1 = 0.83)
  - Performed feature importance analysis — top predictors were
    `Total day minutes`, `Customer service calls`, and `Total day charge`
- **Results**: Accuracy = 95.4% | F1-score = 0.832 | CV F1 = 0.830 ± 0.027
- **Tools**: Python, scikit-learn, pandas, matplotlib

#### Task 3: Neural Networks with TensorFlow/Keras
- **Dataset**: Iris Flower Dataset
- **What was done**:
  - Designed a 3-layer feed-forward neural network:
    - Input layer (4 features) → Hidden layer 1 (16 neurons, ReLU) →
      Hidden layer 2 (8 neurons, ReLU) → Output layer (3 neurons, Softmax)
  - Trained for 100 epochs using Adam optimizer and categorical cross-entropy loss
  - Visualized training vs. validation loss and accuracy curves — no overfitting observed
- **Results**: Test Accuracy = 93.3%
- **Tools**: Python, TensorFlow/Keras, pandas, matplotlib

---

## 📊 Results Summary

| Level | Task | Dataset | Key Metric | Result |
|-------|------|---------|-----------|--------|
| Basic | Linear Regression | House Prices | R² Score | 0.669 |
| Basic | Data Preprocessing | Stock Prices | — | ✅ Complete |
| Intermediate | Logistic Regression | Customer Churn | AUC | 0.814 |
| Intermediate | Decision Tree | Iris | Accuracy | 93.3% |
| Advanced | Random Forest | Customer Churn | Accuracy | 95.4% |
| Advanced | Neural Network | Iris | Accuracy | 93.3% |

---

## ⚙️ How to Run

```bash
# Install dependencies
pip install pandas scikit-learn matplotlib seaborn tensorflow

# Run any task from its level folder
cd level1
python task1_data_preprocessing.py
python task2_linear_regression.py

cd ../level2
python task1_logistic_regression.py
python task2_decision_tree.py

cd ../level3
python task1_random_forest.py
python task3_neural_network.py
```

> **Note**: All scripts read data from the `../data/` folder.
> Make sure the `data/` directory is present before running.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| pandas | Data loading and manipulation |
| scikit-learn | ML models, preprocessing, evaluation |
| TensorFlow / Keras | Deep learning (neural network) |
| matplotlib | Data visualization and plots |
| seaborn | Statistical visualizations |

---

## 🏢 About the Internship

**Organization**: Codveda Technologies  
**Domain**: Machine Learning  
**Duration**: June 19, 2026 – July 19, 2026  
**Mode**: Remote | Flexible Hours  

Codveda Technologies specializes in web development, app development,
digital marketing, SEO optimization, AI/ML automation, and data analysis.

🔗 LinkedIn: [@Codveda Technologies](https://www.linkedin.com/company/codveda)  
📧 Email: support@codveda.com  
🌐 Website: [www.codveda.com](https://www.codveda.com)

---

## 👩‍💻 Author

**Aarti Yadav**  
Machine Learning Intern — Codveda Technologies  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://www.github.com)

---

*Made with ❤️ during the Codveda Technologies ML Internship, 2026*
