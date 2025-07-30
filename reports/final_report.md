### **Final Report: Fraud Detection Project**
#### **10 Academy Week 8&9 Challenge - Adey Innovations Inc.**

**Prepared by: [Your Name]**  
**Date: July 30, 2025, 06:30 AM PDT**  
**GitHub Repository: https://github.com/Yihenew21/Fraud-Detection-ML**

---

### **1. Introduction and Business Context**
This report synthesizes the end-to-end development of fraud detection models for e-commerce and bank transactions at Adey Innovations Inc., addressing significant financial losses due to fraudulent activities. The project, executed over Weeks 8 and 9 of the 10 Academy Challenge, leverages machine learning to identify fraud patterns in two datasets: e-commerce transactions (151,112 samples, 9.36% fraud) and credit card transactions (283,726 samples, 0.17% fraud). Starting with data preprocessing and exploratory analysis (Task 1), it progresses to model building and evaluation (Task 2), and culminates in model explainability using SHAP (Task 3). The solution integrates geolocation and temporal analysis, tackles severe class imbalance with SMOTE, and provides actionable insights to enhance fraud prevention strategies. This narrative connects the business problem—mitigating financial losses—to the technical challenges and the proposed solution, ensuring a seamless flow from problem definition to implementation.

---

### **2. Technical Implementation**

#### **2.1 Data Cleaning and Preprocessing (Task 1)**
- **Datasets**: Loaded `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` using `src/data_utils.py`.
- **Cleaning**: Removed duplicates and imputed missing values (median for `purchase_value` and `Amount`, mode for `source`, `browser`, `sex`).
- **Type Conversion**: Converted `signup_time` and `purchase_time` to datetime for temporal analysis.
- **Geolocation**: Merged IP-to-country data, marking unmatched IPs as 'Unknown'.
- **Output**: Saved processed files to `data/processed/`.
- **Rationale**: These steps ensure data integrity and enable geolocation-based fraud insights, critical for downstream modeling.

#### **2.2 Exploratory Data Analysis (EDA) (Task 1)**
- **Univariate Analysis**: Histograms (`purchase_value_hist.png`, `Amount`) revealed right-skewed distributions, with high values linked to fraud.
  [IMAGE: purchase_value_hist.png]
- **Bivariate Analysis**: Time series (`time_series.png`) showed transaction counts by hour (0-23), with fraud cases spiking at midnight to 4 AM.
  [IMAGE: time_series.png]
  Fraud distribution by top 10 countries (`country_distribution.png`) highlights fraud hotspots with percentage labels.
  [IMAGE: country_distribution.png]
- **Insights**: High transaction values and odd-hour activity are strong fraud indicators, supported by geographic patterns.
- **Implementation**: Conducted in `notebooks/EDA.ipynb` with plots saved to `plots/`.

#### **2.3 Feature Engineering (Task 1)**
- **Time-Based Features** (`src/feature_engineering.py`): 
  - `time_since_signup`: Hours between signup and purchase (hypothesis: rapid purchases by fraudsters).
  - `hour_of_day` and `day_of_week`: Capture temporal fraud patterns.
- **Transaction Frequency**: `trans_freq` per `user_id` (currently 1 due to unique IDs; to be revisited with fuller data).
- **Output**: Saved to `processed_ecommerce_with_features.csv`.
- **Rationale**: These features enhance model ability to detect behavioral and temporal fraud signals.

#### **2.4 Model Building and Training (Task 2)**
- **Preprocessing**: Applied SMOTE (e-commerce: 109,568 each; credit card: 226,602 each), normalized with `StandardScaler`, and encoded categoricals with `OneHotEncoder` in `notebooks/model_training.ipynb`.
- **Models**: Trained Logistic Regression (`random_state=42`, `max_iter=1000`) and Random Forest (`n_estimators=50`) using `src/model_utils.py` with GridSearchCV (`C=[1, 10]` for LR, `n_estimators=[50], max_depth=[20]` for RF).
- **Metrics**:
  - **E-commerce**: RF (F1 0.683, ROC-AUC 0.763, AUC-PR 0.624), LR (F1 0.270, ROC-AUC 0.668, AUC-PR 0.408); CV F1: RF (0.808), LR (0.687).
  - **Credit Card**: RF (F1 0.800, ROC-AUC 0.889, AUC-PR 0.794), LR (F1 0.100, ROC-AUC 0.924, AUC-PR 0.715); CV F1: RF (0.999), LR (0.946).
- **Visualization**: ROC/PR curves in `plots/` (e.g., `ecommerce_roc_curve.png`).
  [IMAGE: ecommerce_roc_curve.png]
  [IMAGE: creditcard_pr_curve.png]
- **Rigor**: 5-fold cross-validation and confusion matrices ensured robust evaluation.

#### **2.5 Model Explainability with SHAP (Task 3)**
- **Implementation**: Used `shap.TreeExplainer` in `src/model_utils.py` to compute SHAP values on 1000-sample subsets, generating Summary Plots (`ecommerce_shap_summary.png`, `creditcard_shap_summary.png`) and Force Plots (`ecommerce_shap_force_0.png`, `creditcard_shap_force_0.png`).
- **Process**: Preprocessed data in `notebooks/model_explainability.ipynb`, balancing with SMOTE and splitting 80-20.
- **Rationale**: SHAP provides global and local interpretability, critical for trust and decision-making.

---

### **3. Insights and Business Value**

#### **3.1 E-commerce Dataset Insights**
- **SHAP Summary Plot**: `purchase_value` and `time_since_signup` are top fraud drivers, with high values and new accounts (low `time_since_signup`) increasing risk. `country_US` highlights regional fraud risks.
  [IMAGE: ecommerce_shap_summary.png]
- **SHAP Force Plot (Sample 0)**: Base value (~0.1) rises to 0.85 due to large `purchase_value` and low `time_since_signup`, confirming these as key local indicators.
- **Business Value**: Focus fraud detection on new accounts with high-value transactions, especially from the U.S., to reduce losses. Implement real-time monitoring for rapid post-signup activity.

#### **3.2 Credit Card Dataset Insights**
- **SHAP Summary Plot**: `V14`, `V17`, and `Amount` dominate, with high `V14` and large `Amount` signaling fraud, while low `V17` may mitigate it.
- **SHAP Force Plot (Sample 0)**: Base value (~0.02) increases to 0.92 due to `V14` and `Amount`, validating these as local drivers.
  [IMAGE: creditcard_shap_force_0.png]
- **Business Value**: Prioritize flagging large transactions with anomalous `V14` values for manual review. Use `V17` as a risk reduction signal to optimize resource allocation.

#### **3.3 Overall Interpretation**
- The Random Forest model, selected for its superior F1 (0.683 e-commerce, 0.800 credit card), ROC-AUC, and AUC-PR, effectively captures feature interactions (e.g., transaction size, account age). SHAP analysis aligns with domain knowledge, where unusual transactions from new or atypical accounts indicate fraud. These insights enable Adey Innovations to:
  - Deploy targeted fraud alerts for high-risk transactions.
  - Adjust credit limits or block accounts based on `time_since_signup` and `V14`.
  - Enhance profitability by reducing false positives through `V17` analysis.

---

### **4. Conclusion and Next Steps**
This project delivers a robust fraud detection framework, from data preprocessing to explainable modeling, addressing Adey Innovations’ financial loss challenge. The Random Forest model, supported by SHAP, provides a balanced and interpretable solution for imbalanced data. Next steps include:
- Final submission pending instructor approval for post-deadline work (July 30, 2025).
- Hyperparameter tuning (e.g., expanding `n_estimators`, `max_depth`) and adding features (e.g., `device_id`) to boost performance.

