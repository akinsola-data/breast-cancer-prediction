# Breast Cancer Prediction

Machine learning classification project predicting whether a 
tumour is malignant or benign using Python and scikit-learn.

## Overview

Built a complete ML pipeline on the UCI Breast Cancer dataset —
from exploratory analysis to trained, evaluated, and compared 
classification models. The target variable is diagnosis: 
0 = malignant (cancerous), 1 = benign (non-cancerous).
This project was completed independently as a portfolio exercise.

## Dataset

- Source: sklearn.datasets load_breast_cancer (UCI ML Repository)
- 569 patients · 30 features · no missing values
- Target: diagnosis (binary — 0 = malignant, 1 = benign)
- Class distribution: 357 benign (63%), 212 malignant (37%)

## Methodology

- Exploratory data analysis — shape, distributions, class balance
- Verified no missing values across all 30 features
- No encoding required — all features are continuous numerical
- Train/test split: 80/20 (455 training, 114 test samples)
- Three models trained: Logistic Regression, Decision Tree, Random Forest
- 5-fold cross-validation for reliable evaluation
- Overfitting detection via training accuracy vs CV mean gap

## Results

| Model | Test Accuracy | CV Mean | CV Std | Overfit |
|---|---|---|---|---|
| Logistic Regression | 95.61% | 95.08% | 1.80% | No |
| Decision Tree | 94.74% | 91.73% | 2.42% | YES |
| Random Forest | 96.49% | 95.61% | 2.28% | No |

**Best model: Random Forest — 95.61% CV Mean Accuracy**

## Overfitting Analysis

Decision Tree achieved 100% training accuracy but only 91.73% 
CV mean — a clear sign of overfitting. The model memorised 
the training data rather than learning generalisable patterns.

Random Forest also achieved 100% training accuracy but maintained 
95.61% CV mean. The ensemble of 100 trees cancels out individual 
memorisation, producing a model that generalises despite fitting 
the training data perfectly.

## Confusion Matrix — Random Forest (Best Model)

| | Predicted: Malignant | Predicted: Benign |
|---|---|---|
| Actual: Malignant | 40 (TN) | 3 (FP) |
| Actual: Benign | 1 (FN) | 70 (TP) |

- Precision (Benign): 0.96
- Recall (Benign): 0.99 — of all patients who were actually 
  benign, the model correctly identified 99% of them
- Recall (Malignant): 0.93 — the model missed 7% of actual 
  cancer cases, classifying them as benign (False Negatives)
- F1 Score: 0.97

## Feature Importance — Random Forest

| Rank | Feature | Importance |
|---|---|---|
| 1 | worst area | 15.39% |
| 2 | worst concave points | 14.47% |
| 3 | mean concave points | 10.62% |
| 4 | worst radius | 7.80% |
| 5 | mean concavity | 6.80% |

The two most predictive features — worst area and worst concave 
points — both describe the most extreme measurements recorded 
for each tumour, suggesting that the severity of the worst 
observed characteristics drives diagnosis more than average values.

## Key Findings

- Random Forest achieved 96.49% test accuracy and 95.61% CV 
  mean with no overfitting detected
- Decision Tree overfitted significantly — 100% train accuracy 
  vs 91.73% CV mean — confirming that single trees memorise 
  rather than learn
- The model correctly identified 99% of benign cases but missed 
  7% of malignant cases — in a clinical context, minimising 
  False Negatives is the priority
- Tumour size and shape irregularity at worst measurements are 
  stronger predictors than average measurements across all features

## Skills Demonstrated

Python · Pandas · Scikit-learn · Matplotlib · Seaborn
Exploratory Data Analysis · Class Balance Analysis
Logistic Regression · Decision Tree · Random Forest
Cross-Validation · Overfitting Detection · Confusion Matrix
Feature Importance Analysis · Medical Data Interpretation

## Tools

Python 3 · Google Colab · scikit-learn · Pandas · Seaborn

## Author

Akinsola Emmanuel  
Statistics Student — University of Ibadan, Nigeria  
[github.com/akinsola-data](https://github.com/akinsola-data)
