# Interim-3 Report: Fraud Detection Project

## Overview
This report summarizes Task 3 (Model Explainability) for the 10 Academy Week 8&9 Challenge, building on Tasks 1 and 2. It uses SHAP (Shapley Additive exPlanations) to interpret the best-performing Random Forest model for fraud detection on e-commerce and credit card datasets at Adey Innovations Inc. The work is implemented in `notebooks/interim_3.ipynb` with modular functions in `src/`. Note: This report was finalized post-deadline on July 30, 2025, due to SHAP debugging; resubmission is subject to instructor approval.

## Data Preprocessing
- **Datasets**: Used `processed_ecommerce_with_features.csv` (151,112 samples) and `processed_creditcard.csv` (283,726 samples) from Task 1.
- **Features**: E-commerce (`purchase_value`, `time_since_signup`, `hour_of_day`, `day_of_week`, `trans_freq` with categorical `source`, `browser`, `country`); Credit Card (`Time`, `Amount`, `V1`–`V28`).
- **Processing**: Applied SMOTE to balance classes, normalized numerical features with `StandardScaler`, and encoded e-commerce categoricals with `OneHotEncoder`. Justification: SMOTE prioritizes recall to minimize missed fraud cases.
- **Train-Test Split**: 80-20 split with stratification.

## Model Training and Evaluation
- **Models**: Trained Logistic Regression and Random Forest on both datasets (results from Task 2).
- **E-commerce Metrics**: Random Forest: F1 0.683, ROC-AUC 0.763, AUC-PR 0.624; Logistic Regression: F1 0.270, ROC-AUC 0.668, AUC-PR 0.408.
- **Credit Card Metrics**: Random Forest: F1 0.800, ROC-AUC 0.889, AUC-PR 0.794; Logistic Regression: F1 0.100, ROC-AUC 0.924, AUC-PR 0.715.
- **Best Model**: Random Forest, due to higher F1, ROC-AUC, and AUC-PR, balancing precision and recall for imbalanced data.

## Visualization
- **SHAP Plots**: Generated `ecommerce_shap_summary.png`, `creditcard_shap_summary.png`, `ecommerce_shap_force_0.png`, `creditcard_shap_force_0.png` using `src/shap_utils.py`.

## Model Explainability with SHAP
- **E-commerce Dataset**: 
  - **Summary Plot (`ecommerce_shap_summary.png`)**: The plot reveals `purchase_value` and `time_since_signup` as the strongest predictors of fraud, with high transaction values (red dots on the positive SHAP side) and low account ages (blue dots on the positive side) increasing fraud likelihood. The categorical feature `country_US` also shows significance, suggesting specific regions contribute to fraud risk.
  - **Force Plot (`ecommerce_shap_force_0.png`)**: For the first sample, the base value (approximately 0.1 or 10% fraud probability) is pushed to a final prediction of 0.85 (85%) due to positive contributions from `time_since_signup` (indicating a new account) and `purchase_value` (a large transaction), highlighting these as key local drivers.
- **Credit Card Dataset**: 
  - **Summary Plot (`creditcard_shap_summary.png`)**: The plot identifies `V14`, `V17`, and `Amount` as top contributors, with high `V14` values (red) and large transaction amounts (red) strongly associated with fraud, while low `V17` values (blue) may reduce it.
  - **Force Plot (`creditcard_shap_force_0.png`)**: For the first sample, the base value (approximately 0.02 or 2%) rises to 0.92 (92%) due to significant positive contributions from `V14` and `Amount`, indicating these features drive the fraud classification locally.
- **Interpretation**: SHAP analysis demonstrates that the Random Forest model effectively captures nuanced interactions between transaction size (`purchase_value`, `Amount`), account age (`time_since_signup`), and derived features (`V14`, `V17`). These align with domain knowledge, where large, unusual transactions from new or atypical accounts signal fraud. The global importance from Summary Plots and local insights from Force Plots enhance model interpretability, providing actionable indicators for fraud prevention at Adey Innovations Inc.

## Repository
- GitHub: `https://github.com/Yihenew21/Fraud-Detection-ML`
- Structure: Updated with `notebooks/interim_3.ipynb`, `src/shap_utils.py`, `plots/`, and `interim_3_report.md`.

## Next Steps
- Finalize and submit the project, pending instructor feedback on post-deadline submission.
- Consider hyperparameter tuning or additional features (e.g., device ID) to further improve model performance.

## Date
- Report completed: July 30, 2025