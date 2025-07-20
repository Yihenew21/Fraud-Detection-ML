# Interim-2 Report: Fraud Detection Project

## Overview
This report summarizes Task 2 (Model Building and Training) for the 10 Academy Week 8&9 Challenge, focusing on developing Logistic Regression and Random Forest models for fraud detection on e-commerce and credit card datasets at Adey Innovations Inc. The work is implemented in `notebooks/interim_2.ipynb`.

## Data Preprocessing
- **Datasets**: Used `processed_ecommerce_with_features.csv` (151,112 samples) and `processed_creditcard.csv` (283,726 samples) from Task 1.
- **Features**: E-commerce (`purchase_value`, `time_since_signup`, etc., with categorical `source`, `browser`, `country`); Credit Card (`Time`, `Amount`, `V1`–`V28`).
- **Processing**: Applied SMOTE to balance classes (e-commerce: 109,568 each; credit card: 226,602 each), normalized numerical features with `StandardScaler`, and encoded e-commerce categoricals with `OneHotEncoder`. Justification: SMOTE prioritizes recall to minimize missed fraud cases.
- **Train-Test Split**: 80-20 split with stratification.

## Model Training and Evaluation
- **Models**: Trained Logistic Regression (`random_state=42`, `max_iter=1000`) and Random Forest (`n_estimators=50`) on both datasets.
- **E-commerce Metrics**:
  - Logistic Regression: Accuracy 0.651, Precision 0.168, Recall 0.690, F1 0.270, ROC-AUC 0.668, AUC-PR 0.408
  - Random Forest: Accuracy 0.954, Precision 0.968, Recall 0.528, F1 0.683, ROC-AUC 0.763, AUC-PR 0.624
  - CV F1: LR (0.687), RF (0.808)
- **Credit Card Metrics**:
  - Logistic Regression: Accuracy 0.974, Precision 0.053, Recall 0.874, F1 0.100, ROC-AUC 0.924, AUC-PR 0.715
  - Random Forest: Accuracy 0.999, Precision 0.822, Recall 0.779, F1 0.800, ROC-AUC 0.889, AUC-PR 0.794
  - CV F1: LR (0.946), RF (0.999)
- **Insights**: Random Forest outperforms Logistic Regression on both datasets with higher F1, ROC-AUC, and AUC-PR, indicating better balance for imbalanced fraud detection. Logistic Regression’s high recall on credit card data is notable but comes with low precision.

## Visualization
- **ROC/PR Curves**: Plotted in `plots/ecommerce_roc_curve.png`, `plots/ecommerce_pr_curve.png`, `plots/creditcard_roc_curve.png`, `plots/creditcard_pr_curve.png`.

## Model Comparison and Justification
- **E-commerce Dataset**: Random Forest shows a higher F1-score (0.683) and AUC-PR (0.624) compared to Logistic Regression (F1 0.270, AUC-PR 0.408), indicating a better balance of precision (0.968) and recall (0.528). This makes Random Forest the best model for detecting fraud with fewer false positives, despite Logistic Regression’s higher recall (0.690).
- **Credit Card Dataset**: Random Forest achieves a higher F1-score (0.800) and AUC-PR (0.794) than Logistic Regression (F1 0.100, AUC-PR 0.715), with a strong recall (0.779) and precision (0.822), compared to Logistic Regression’s recall (0.874) but very low precision (0.053). Random Forest’s superior overall performance (ROC-AUC 0.889 vs. 0.924) and stability (CV F1 0.999 vs. 0.946) make it the best model, though Logistic Regression may be preferred if maximizing recall is the sole priority.
- **Justification**: Random Forest is chosen as the best model across both datasets due to its robustness to feature interactions, higher F1-scores, ROC-AUC, and AUC-PR values, which are critical for imbalanced fraud detection. Its ability to balance precision and recall, combined with excellent cross-validation stability, outweighs Logistic Regression’s simplicity and higher recall in specific cases.

## Repository
- GitHub: `https://github.com/Yihenew21/Fraud-Detection-Adey-Innovations`
- Structure: Updated with `notebooks/model_training.ipynb` and `plots/` files.

## Next Steps
- Finalize submission and address runtime optimization if needed.