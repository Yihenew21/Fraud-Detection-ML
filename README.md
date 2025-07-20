# Fraud Detection Project - 10 Academy Week 8&9

## Overview
This project develops fraud detection models for e-commerce and bank transactions at Adey Innovations Inc. The solution addresses severe class imbalance (9.36% fraud in e-commerce, 0.17% in credit card data), incorporates geolocation analysis, and implements temporal fraud pattern detection through comprehensive data preprocessing and machine learning modeling.

**Key Accomplishments:**
- Comprehensive EDA revealing fraud patterns in transaction values, timing, and geography
- Advanced feature engineering with temporal and behavioral indicators  
- SMOTE implementation for class imbalance handling
- Logistic Regression and Random Forest model comparison
- Performance evaluation prioritizing fraud detection effectiveness

## Business Problem
Fraudulent transactions cause significant financial losses. This project identifies fraud patterns across e-commerce (151,112 samples) and credit card (283,726 samples) datasets and builds detection models optimized for fraud identification while managing false positive rates.

## Setup & Installation

### Prerequisites
- Python 3.11+
- Jupyter Lab/Notebook
- Git

### Quick Start
```bash
# Clone the repository  
git clone https://github.com/Yihenew21/Fraud-Detection-ML.git
cd Fraud-Detection-ML

# Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/MacOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

### Data Setup
Place the following datasets in `data/raw/`:
- `Fraud_Data.csv` - E-commerce transaction data
- `IpAddress_to_Country.csv` - IP geolocation mapping  
- `creditcard.csv` - Credit card transaction data

## Project Architecture

```
fraud-detection/
├── data/
│   ├── raw/                           # Original datasets
│   └── processed/                     # Cleaned datasets with engineered features
├── notebooks/
│   ├── EDA.ipynb                     # Task 1: Data analysis and preprocessing
│   └── model_training.ipynb          # Task 2: Model building and evaluation
├── src/                              # Modular Python scripts
│   ├── data_utils.py                 # Data loading and preprocessing functions
│   └── feature_engineering.py       # Feature creation and transformation
├── plots/                            # Visualization outputs
│   ├── purchase_value_hist.png       # Purchase value distribution analysis
│   ├── time_series.png              # Temporal fraud patterns  
│   ├── country_distribution.png     # Geographic fraud hotspots
│   ├── ecommerce_roc_curve.png      # E-commerce model ROC curves
│   ├── ecommerce_pr_curve.png       # E-commerce precision-recall curves
│   ├── creditcard_roc_curve.png     # Credit card model ROC curves
│   └── creditcard_pr_curve.png      # Credit card precision-recall curves
├── reports/                         # Technical documentation
│   ├── interim_1_report.md          # Task 1: Data analysis findings
│   └── interim_2_report.md          # Task 2: Model evaluation results
└── requirements.txt                 # Project dependencies
```

## Implementation Details

### 🔍 **Task 1: Data Analysis & Preprocessing**
**EDA Findings:**
- **Class Distribution**: E-commerce 90.64% non-fraud, 9.36% fraud; Credit card 99.83% non-fraud, 0.17% fraud
- **Fraud Patterns**: High purchase values correlate with fraud; consistent fraud timing across all hours indicates systematic attacks
- **Geographic Insights**: Fraud hotspots identified with China (11.6%), Canada (13.8%), and Unknown locations (9.1%) showing elevated fraud rates
- **Temporal Analysis**: Fraud occurs consistently (~500-700 cases/hour) while legitimate transactions peak during business hours

**Feature Engineering:**
- **`time_since_signup`**: Duration between signup and purchase (hypothesis: fraudsters make rapid purchases)
- **`hour_of_day`**: Purchase timing for circadian pattern analysis
- **`day_of_week`**: Weekly fraud pattern identification
- **Geographic mapping**: IP-to-country conversion for location-based risk assessment

**Data Processing:**
- Missing value imputation: Median for numerical, mode for categorical
- Duplicate removal and data type optimization
- Geolocation enhancement through IP-country mapping

### 🤖 **Task 2: Model Building & Training**
**Preprocessing Pipeline:**
- **SMOTE Application**: Balanced classes (E-commerce: 109,568 each; Credit card: 226,602 each)
- **Normalization**: StandardScaler for numerical features
- **Encoding**: OneHotEncoder for e-commerce categorical features
- **Data Split**: 80-20 train-test with stratification

**Model Implementation:**
- **Logistic Regression**: `random_state=42`, `max_iter=1000`
- **Random Forest**: `n_estimators=50`
- **Cross-Validation**: F1-score evaluation for model stability

## Model Performance Results

### E-commerce Dataset Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | AUC-PR | CV F1 |
|-------|----------|-----------|--------|----------|---------|--------|-------|
| Logistic Regression | 0.651 | 0.168 | 0.690 | 0.270 | 0.668 | 0.408 | 0.687 |
| Random Forest | 0.954 | 0.968 | 0.528 | 0.683 | 0.763 | 0.624 | 0.808 |

### Credit Card Dataset Performance  
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | AUC-PR | CV F1 |
|-------|----------|-----------|--------|----------|---------|--------|-------|
| Logistic Regression | 0.974 | 0.053 | 0.874 | 0.100 | 0.924 | 0.715 | 0.946 |
| Random Forest | 0.999 | 0.822 | 0.779 | 0.800 | 0.889 | 0.794 | 0.999 |

## Running the Analysis

### Task 1: Data Analysis & Preprocessing
```bash
# Open and run: notebooks/EDA.ipynb
```
**Outputs:**
- `data/processed/processed_ecommerce_with_features.csv`
- `data/processed/processed_creditcard.csv`
- EDA visualizations in `plots/`

### Task 2: Model Training & Evaluation
```bash  
# Open and run: notebooks/model_training.ipynb
```
**Outputs:**
- Model performance metrics and comparisons
- ROC and Precision-Recall curves in `plots/`
- Trained model evaluation results

## Key Findings

### **Model Selection Results**
- **E-commerce**: Random Forest selected (F1: 0.683, AUC-PR: 0.624) for superior precision-recall balance
- **Credit Card**: Random Forest selected (F1: 0.800, AUC-PR: 0.794) for robust overall performance
- **Rationale**: Random Forest's ability to handle feature interactions and maintain precision-recall balance outweighs Logistic Regression's higher recall in specific cases

### **Business Insights**
- **Fraud Indicators**: High transaction values and geographic location are key fraud predictors
- **Temporal Patterns**: Systematic fraud activity suggests automated attacks rather than opportunistic fraud
- **Class Imbalance**: SMOTE successfully addresses severe imbalance while maintaining test data integrity

## Code Quality & Testing

### **Modular Design**
- Reusable functions in `src/` directory
- Comprehensive inline documentation
- Consistent processing pipelines with reproducible results

### **Validation Approach**
- Stratified train-test splits maintain class distribution
- Cross-validation ensures model stability
- Fraud-focused evaluation metrics (F1, ROC-AUC, AUC-PR)

## Dependencies
Key requirements include:
- pandas, numpy for data manipulation
- scikit-learn for modeling and preprocessing
- imbalanced-learn for SMOTE implementation
- matplotlib, seaborn for visualization
- jupyter for notebook environment

## Project Status

### ✅ **Task 1 - Completed**
- Comprehensive EDA with fraud pattern identification
- Feature engineering with business logic justification  
- Data preprocessing pipeline with quality assurance
- Class imbalance analysis and strategy development

### ✅ **Task 2 - Completed**
- SMOTE implementation for class balance
- Logistic Regression and Random Forest training
- Comprehensive model evaluation and comparison
- Performance visualization with ROC and PR curves
- Model selection based on fraud detection effectiveness

## Repository
- **GitHub**: `https://github.com/Yihenew21/Fraud-Detection-Adey-Innovations`
- **Documentation**: Complete technical reports in `reports/` directory
- **Reproducibility**: All analysis steps documented and executable

---
*This project demonstrates end-to-end fraud detection implementation from data preprocessing through model evaluation and selection.*