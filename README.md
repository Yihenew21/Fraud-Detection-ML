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

**Note**: Ensure datasets are placed in the correct directory structure before running the notebooks to avoid path errors.

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

**Step-by-Step Execution:**
1. **Data Loading**: Execute the data loading section using `src/data_utils.py`
2. **Data Cleaning**: Run the data cleaning section for duplicate removal and missing value imputation
3. **Exploratory Analysis**: Execute the EDA section for univariate and bivariate analysis
4. **Feature Engineering**: Run the feature engineering section to create temporal and behavioral features
5. **Class Imbalance Analysis**: Execute the class imbalance analysis section

**Outputs:**
- `data/processed/processed_ecommerce_with_features.csv`
- `data/processed/processed_creditcard.csv`
- EDA visualizations in `plots/`

### Task 2: Model Training & Evaluation
```bash  
# Open and run: notebooks/model_training.ipynb
```

**Step-by-Step Execution:**
1. **Data Preprocessing**: Execute the preprocessing section to apply SMOTE, normalization, and encoding
2. **Train-Test Split**: Run the data splitting section for stratified data splitting
3. **Model Training**: Execute the model training section for Logistic Regression and Random Forest
4. **Model Evaluation**: Run the evaluation section for performance metrics calculation
5. **Visualization**: Execute the visualization section to generate ROC and PR curves

**Outputs:**
- Model performance metrics and comparisons
- ROC and Precision-Recall curves in `plots/`
- Trained model evaluation results

## Code Structure & Documentation

### **Modular Design Principles**
The project follows clean code principles with modular functions:

- **`src/data_utils.py`**: Contains reusable data loading and preprocessing functions
  - `load_data()`: Standardized data loading with error handling
  - `clean_data()`: Comprehensive data cleaning pipeline
  - `merge_geolocation()`: IP-to-country mapping functionality

- **`src/feature_engineering.py`**: Dedicated feature creation module
  - `create_time_features()`: Temporal feature extraction
  - `calculate_transaction_frequency()`: Behavioral pattern analysis
  - `encode_categorical_features()`: Standardized encoding pipeline

### **Code Quality Standards**
- **Inline Documentation**: Each function includes detailed docstrings explaining parameters, returns, and business logic
- **Error Handling**: Comprehensive try-catch blocks for data loading and processing steps
- **Reproducibility**: Fixed random seeds (`random_state=42`) ensure consistent results
- **Performance Optimization**: Efficient pandas operations and vectorized computations

### **Notebook Structure & Flow**
Both notebooks follow a logical progression:

1. **Setup Phase**: Import statements and environment configuration
2. **Data Preparation**: Loading, cleaning, and initial exploration
3. **Analysis/Modeling**: Core functionality with detailed explanations
4. **Evaluation**: Results interpretation and business insights
5. **Output Generation**: Saving processed data and visualizations

## Key Findings

### **Model Selection Results**
- **E-commerce**: Random Forest selected (F1: 0.683, AUC-PR: 0.624) for superior precision-recall balance
- **Credit Card**: Random Forest selected (F1: 0.800, AUC-PR: 0.794) for robust overall performance
- **Rationale**: Random Forest's ability to handle feature interactions and maintain precision-recall balance outweighs Logistic Regression's higher recall in specific cases

### **Business Insights**
- **Fraud Indicators**: High transaction values and geographic location are key fraud predictors
- **Temporal Patterns**: Systematic fraud activity suggests automated attacks rather than opportunistic fraud
- **Class Imbalance**: SMOTE successfully addresses severe imbalance while maintaining test data integrity

### **Class Imbalance Impact Analysis**
The severe class imbalance in both datasets has significant implications:

- **E-commerce Dataset**: 9.36% fraud rate requires careful balance between precision and recall
  - **Without SMOTE**: Models would achieve high accuracy by predicting all transactions as legitimate
  - **With SMOTE**: Balanced training enables proper fraud pattern learning
  - **Business Impact**: Missing 1 fraud case costs significantly more than 1 false positive

- **Credit Card Dataset**: 0.17% fraud rate represents extreme imbalance
  - **Challenge**: Only 492 fraud cases out of 284,807 total transactions
  - **SMOTE Strategy**: Synthetic oversampling creates balanced 50-50 training distribution
  - **Validation**: Original test set proportions maintained for realistic performance evaluation
  - **Result**: Models learn fraud patterns effectively while maintaining real-world applicability

## Code Quality & Testing

### **Validation Approach**
- Stratified train-test splits maintain class distribution
- Cross-validation ensures model stability
- Fraud-focused evaluation metrics (F1, ROC-AUC, AUC-PR)

### **Error Handling & Robustness**
- File existence validation before data loading
- Missing value detection and appropriate handling strategies
- Data type validation and conversion safeguards
- Memory usage optimization for large datasets

## Troubleshooting Guide

### **Common Setup Issues**
1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **File Path Errors**: Verify datasets are placed in `data/raw/` directory
3. **Memory Issues**: For large datasets, consider increasing available RAM or using data sampling
4. **Jupyter Kernel Issues**: Restart kernel and rerun from the beginning if encountering state issues

### **Data Processing Issues**
1. **Missing Files**: Check that all three datasets are present and correctly named
2. **Encoding Issues**: Ensure CSV files use UTF-8 encoding
3. **Memory Warnings**: Normal for large datasets; processing continues automatically

## Dependencies
Key requirements include:
- pandas, numpy for data manipulation
- scikit-learn for modeling and preprocessing
- imbalanced-learn for SMOTE implementation
- matplotlib, seaborn for visualization
- jupyter for notebook environment

**Full dependency list available in `requirements.txt`**

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

## Contributing & Code Standards

### **Development Workflow**
1. Create feature branches for new development
2. Write comprehensive inline comments explaining business logic
3. Include docstrings for all functions
4. Test code with sample data before committing
5. Update documentation for any new features

### **Documentation Standards**
- All functions must include docstrings with parameter descriptions
- Complex business logic requires inline comments explaining rationale
- README updates required for any structural changes
- Report files maintained for technical documentation

## Repository
- **GitHub**: `https://github.com/Yihenew21/Fraud-Detection-Adey-Innovations`
- **Documentation**: Complete technical reports in `reports/` directory
- **Reproducibility**: All analysis steps documented and executable
- **Code Quality**: Modular design with comprehensive documentation

---
*This project demonstrates end-to-end fraud detection implementation from data preprocessing through model evaluation and selection, with emphasis on code quality, documentation clarity, and business impact analysis.*