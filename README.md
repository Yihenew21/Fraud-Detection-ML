# Fraud Detection Project - 10 Academy Week 8&9

## Overview
This project develops fraud detection models for e-commerce and bank transactions at Adey Innovations Inc. The solution addresses severe class imbalance (9.36% fraud in e-commerce, 0.17% in credit card data), incorporates geolocation analysis, and implements temporal fraud pattern detection through comprehensive data preprocessing and machine learning modeling. The project culminates in Task 3 with SHAP (Shapley Additive exPlanations) to interpret the best-performing Random Forest models, providing actionable insights into fraud drivers.

**Key Accomplishments:**
- Comprehensive EDA revealing fraud patterns in transaction values, timing, and geography
- Advanced feature engineering with temporal and behavioral indicators
- SMOTE implementation for class imbalance handling
- Logistic Regression and Random Forest model comparison
- Performance evaluation prioritizing fraud detection effectiveness
- SHAP-based model explainability with Summary and Force plots for global and local feature importance

## Business Problem
Fraudulent transactions cause significant financial losses. This project identifies fraud patterns across e-commerce (151,112 samples) and credit card (283,726 samples) datasets and builds detection models optimized for fraud identification while managing false positive rates. SHAP analysis enhances interpretability, aiding business decisions on fraud prevention.

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
Place the following datasets in `data/raw/` with the specified formats and sizes:
- `Fraud_Data.csv`: E-commerce transaction data, CSV format, ~151,112 rows, including columns `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `source`, `browser`, `sex`, `ip_address`, `class`.
- `IpAddress_to_Country.csv`: IP geolocation mapping, CSV format, mapping IP ranges to countries.
- `creditcard.csv`: Credit card transaction data, CSV format, ~283,726 rows, including columns `Time`, `Amount`, `V1`-`V28`, `Class`.

**Note**: Ensure datasets are placed in the correct directory structure before running the notebooks to avoid path errors. Verify file integrity and encoding (UTF-8) to prevent loading issues.

## Project Architecture
```
fraud-detection/
├── data/
│   ├── raw/                          # Original datasets
│   └── processed/                    # Cleaned datasets with engineered features
├── notebooks/
│   ├── EDA.ipynb                     # Task 1: Data analysis and preprocessing
│   ├── model_training.ipynb          # Task 2: Model building and evaluation
│   └── model_explainability.ipynb    # Task 3: Model explainability with SHAP
├── src/
│   ├── data_utils.py                 # Data loading and preprocessing functions
│   ├── model_utils.py                # Model training, evaluation, and SHAP computation functions
│   ├── feature_engineering.py        # Feature creation and transformation
│   └── shap_utils.py                 # SHAP analysis functions
├── plots/
│   ├── purchase_value_hist.png       # Purchase value distribution analysis
│   ├── time_series.png               # Temporal fraud patterns
│   ├── country_distribution.png      # Geographic fraud hotspots
│   ├── ecommerce_roc_curve.png       # E-commerce model ROC curves
│   ├── ecommerce_pr_curve.png        # E-commerce precision-recall curves
│   ├── creditcard_roc_curve.png      # Credit card model ROC curves
│   ├── creditcard_pr_curve.png       # Credit card precision-recall curves
│   ├── ecommerce_shap_summary.png    # E-commerce SHAP Summary Plot
│   ├── creditcard_shap_summary.png   # Credit card SHAP Summary Plot
│   ├── ecommerce_shap_force_0.png    # E-commerce SHAP Force Plot (Sample 0)
│   └── creditcard_shap_force_0.png   # Credit card SHAP Force Plot (Sample 0)
├── reports/
│   ├── interim_1_report.md           # Task 1: Data analysis findings
│   ├── interim_2_report.md           # Task 2: Model evaluation results
│   └── interim_3_report.md           # Task 3: Model explainability findings
└── requirements.txt                  # Project dependencies
```

## Implementation Details

### 🔍 Task 1: Data Analysis & Preprocessing
#### EDA Findings
- **Class Distribution**: E-commerce: 90.64% non-fraud, 9.36% fraud; Credit card: 99.83% non-fraud, 0.17% fraud.
- **Fraud Patterns**: High purchase values correlate with fraud; consistent fraud timing across all hours indicates systematic attacks.
- **Geographic Insights**: Fraud hotspots identified in China (11.6%), Canada (13.8%), and Unknown locations (9.1%) show elevated fraud rates.
- **Temporal Analysis**: Fraud occurs consistently (~500-700 cases/hour) while legitimate transactions peak during business hours.

#### Feature Engineering
- `time_since_signup`: Duration between signup and purchase (hypothesis: fraudsters make rapid purchases).
- `hour_of_day`: Purchase timing for circadian pattern analysis.
- `day_of_week`: Weekly fraud pattern identification.
- **Geographic mapping**: IP-to-country conversion for location-based risk assessment.

#### Data Processing
- Missing value imputation: Median for numerical, mode for categorical.
- Duplicate removal and data type optimization.
- Geolocation enhancement through IP-country mapping.

### 🤖 Task 2: Model Building & Training
#### Preprocessing Pipeline
- **SMOTE Application**: Balanced classes (E-commerce: 109,568 each; Credit card: 226,602 each).
- **Normalization**: StandardScaler for numerical features.
- **Encoding**: OneHotEncoder for e-commerce categorical features.
- **Data Split**: 80-20 train-test with stratification.

#### Model Implementation
- **Logistic Regression**: `random_state=42`, `max_iter=1000`.
- **Random Forest**: `n_estimators=50`.
- **Cross-Validation**: F1-score evaluation for model stability.

### 📊 Task 3: Model Explainability with SHAP
#### Implementation
- Used `shap.TreeExplainer` via `model_utils.py` to compute SHAP values for the best Random Forest models on 1000-sample subsets.
- Generated **Summary Plots** (`ecommerce_shap_summary.png`, `creditcard_shap_summary.png`) and **Force Plots** (`ecommerce_shap_force_0.png`, `creditcard_shap_force_0.png`) using `model_utils.py` functions.
- Implemented in `src/model_utils.py` with modular functions (`train_evaluate_model`, `compute_shap_values`, `plot_shap_summary`, `plot_shap_force`).

#### Analysis
- **E-commerce**: Summary Plot shows `purchase_value` and `time_since_signup` as top fraud drivers, with high values and new accounts increasing risk. Force Plot for Sample 0 indicates a base value (~0.1) rising to ~0.85 due to these features.
- **Credit Card**: Summary Plot highlights `V14`, `V17`, and `Amount` as key indicators, with large transactions and high `V14` values linked to fraud. Force Plot for Sample 0 shows a base value (~0.02) increasing to ~0.92.
- **Insights**: SHAP confirms the model captures transaction size, account age, and derived feature interactions, aligning with domain knowledge and enhancing interpretability.

## Model Performance Results

### E-commerce Dataset Performance
| Model             | Accuracy | Precision | Recall | F1-Score | ROC-AUC | AUC-PR | CV F1  |
|-------------------|----------|-----------|--------|----------|---------|--------|--------|
| Logistic Regression | 0.651   | 0.168     | 0.690  | 0.270    | 0.668   | 0.408  | 0.687  |
| Random Forest     | 0.954   | 0.968     | 0.528  | 0.683    | 0.763   | 0.624  | 0.808  |

### Credit Card Dataset Performance
| Model             | Accuracy | Precision | Recall | F1-Score | ROC-AUC | AUC-PR | CV F1  |
|-------------------|----------|-----------|--------|----------|---------|--------|--------|
| Logistic Regression | 0.974   | 0.053     | 0.874  | 0.100    | 0.924   | 0.715  | 0.946  |
| Random Forest     | 0.999   | 0.822     | 0.779  | 0.800    | 0.889   | 0.794  | 0.999  |

## Running the Analysis

### Task 1: Data Analysis & Preprocessing
**To run the analysis, follow these steps:**
1. Open `notebooks/EDA.ipynb` in Jupyter Lab. (Runtime: ~2 minutes)
   ```bash
   jupyter lab notebooks/EDA.ipynb
   ```
2. Execute the "Setup Phase" cells to import libraries. (Runtime: ~30 seconds)
   ```bash
   # Run all cells under "Setup Phase" manually
   ```
3. Execute the "Data Loading" cells using `src/data_utils.py`. (Runtime: ~1 minute)
   ```bash
   # Run cells under "Data Loading" manually
   ```
4. Execute the "Data Cleaning" cells for duplicate removal and imputation. (Runtime: ~30 seconds)
   ```bash
   # Run cells under "Data Cleaning" manually
   ```
5. Execute the "Exploratory Analysis" cells to generate plots. (Runtime: ~1 minute)
   ```bash
   # Run cells under "Exploratory Analysis" manually
   ```
6. Execute the "Feature Engineering" cells to create temporal features. (Runtime: ~1 minute)
   ```bash
   # Run cells under "Feature Engineering" manually
   ```
7. Execute the "Class Imbalance Analysis" cells. (Runtime: ~30 seconds)
   ```bash
   # Run cells under "Class Imbalance Analysis" manually
   ```

**Expected Outputs:**
- `data/processed/processed_ecommerce_with_features.csv`
- `data/processed/processed_creditcard.csv`
- EDA visualizations in `plots/`

### Task 2: Model Training & Evaluation
**To run the analysis, follow these steps:**
1. Open `notebooks/model_training.ipynb` in Jupyter Lab. (Runtime: ~2 minutes)
   ```bash
   jupyter lab notebooks/model_training.ipynb
   ```
2. Execute the "Setup Phase" cells to import libraries. (Runtime: ~30 seconds)
   ```bash
   # Run all cells under "Setup Phase" manually
   ```
3. Execute the "Data Preprocessing" cells to apply SMOTE and normalization. (Runtime: ~2 minutes)
   ```bash
   # Run cells under "Data Preprocessing" manually
   ```
4. Execute the "Train-Test Split" cells. (Runtime: ~1 minute)
   ```bash
   # Run cells under "Train-Test Split" manually
   ```
5. Execute the "Model Training and Evaluation" cells to train Logistic Regression and Random Forest. (Runtime: ~25 minutes)
   ```bash
   # Run cells under "Model Training and Evaluation" manually
   ```
6. Execute the "Visualization" cells to generate ROC and PR curves. (Runtime: ~1 minute)
   ```bash
   # Run cells under "Visualization" manually
   ```

**Expected Outputs:**
- Model performance metrics and comparisons
- ROC and Precision-Recall curves in `plots/`
- Trained model evaluation results

### Task 3: Model Explainability with SHAP
**To run the analysis, follow these steps:**
1. Open `notebooks/model_explainability.ipynb` in Jupyter Lab. (Runtime: ~2 minutes)
   ```bash
   jupyter lab notebooks/model_explainability.ipynb
   ```
2. Execute the "Setup Phase" cells to import libraries. (Runtime: ~30 seconds)
   ```bash
   # Run all cells under "Setup Phase" manually
   ```
3. Execute the "Data Preparation" cells to load preprocessed data. (Runtime: ~1 minute)
   ```bash
   # Run cells under "Data Preparation" manually
   ```
4. Execute the "Train and Evaluate Models" cells to run `compute_shap_values`. (Runtime: ~25 minutes)
   ```bash
   # Run cells under "Train and Evaluate Models" manually
   ```
5. Execute the "SHAP Computation And Plot Generation" cells to create Summary and Force plots. (Runtime: ~2 minutes)
   ```bash
   # Run cells under "SHAP Computation and Plot Generation" manually
   ```
6. Execute the "Interpretation" cells to review and update findings. (Runtime: ~1 minute)
   ```bash
   # Run cells under "Interpretation" manually
   ```

**Expected Outputs:**
- SHAP Summary and Force plots in `plots/`
- Interpretations in `notebooks/model_explainability.ipynb` and `interim_3_report.md`

## Code Structure & Documentation

### Modular Design Principles
The project follows clean code principles with modular functions:
- **`src/data_utils.py`**:
  - `load_data()`: Standardized data loading with error handling
  - `clean_data()`: Comprehensive data cleaning pipeline
  - `merge_geolocation()`: IP-to-country mapping functionality
- **`src/feature_engineering.py`**:
  - `create_time_features()`: Temporal feature extraction
  - `calculate_transaction_frequency()`: Behavioral pattern analysis
  - `encode_categorical_features()`: Standardized encoding pipeline
- **`src/model_utils.py`**:
  - `train_evaluate_model()`: Trains, tunes, and evaluates Logistic Regression and Random Forest models with hyperparameter tuning and metrics
  - `compute_shap_values()`: Computes SHAP values for model interpretation on a subset
  - `plot_shap_summary()`: Generates global SHAP Summary Plots
  - `plot_shap_force()`: Creates local SHAP Force Plots
- **`src/shap_utils.py`**:
  - `compute_shap_values()`: Computes SHAP values for model interpretation
  - `plot_shap_summary()`: Generates global Summary Plots
  - `plot_shap_force()`: Creates local Force Plots

### Code Quality Standards
- **Inline Documentation**: Each function includes detailed docstrings explaining parameters, returns, and business logic.
- **Error Handling**: Comprehensive `try-catch` blocks for data loading and processing steps.
- **Reproducibility**: Fixed random seeds (`random_state=42`) ensure consistent results.
- **Performance Optimization**: Efficient pandas operations and vectorized computations.

### Notebook Structure & Flow
All notebooks follow a logical progression:
1. **Setup Phase**: Import statements and environment configuration.
2. **Data Preparation**: Loading, cleaning, and initial exploration.
3. **Analysis/Modeling**: Core functionality with detailed explanations.
4. **Evaluation**: Results interpretation and business insights.
5. **Output Generation**: Saving processed data and visualizations.

## Key Findings

### Model Selection Results
- **E-commerce**: Random Forest selected (F1: 0.683, AUC-PR: 0.624) for superior precision-recall balance.
- **Credit Card**: Random Forest selected (F1: 0.800, AUC-PR: 0.794) for robust overall performance.
- **Rationale**: Random Forest's ability to handle feature interactions and maintain precision-recall balance outweighs Logistic Regression's higher recall in specific cases.

### Business Insights
- **Fraud Indicators**: High transaction values (`purchase_value`, `Amount`) and geographic location (`country_US`) are key predictors.
- **Temporal Patterns**: Systematic fraud activity suggests automated attacks, with `time_since_signup` and `hour_of_day` highlighting new account risks.
- **SHAP Insights**: `V14` and `V17` (credit card) and derived patterns enhance detection, aligning with domain expertise.
- **Class Imbalance**: SMOTE successfully addresses severe imbalance while maintaining test data integrity.

### Class Imbalance Impact Analysis
- **E-commerce Dataset**: 9.36% fraud rate requires careful balance between precision and recall.
  - **Without SMOTE**: Models would achieve high accuracy by predicting all transactions as legitimate.
  - **With SMOTE**: Balanced training enables proper fraud pattern learning.
  - **Business Impact**: Missing 1 fraud case costs significantly more than 1 false positive.
- **Credit Card Dataset**: 0.17% fraud rate represents extreme imbalance.
  - **Challenge**: Only 492 fraud cases out of 284,807 total transactions.
  - **SMOTE Strategy**: Synthetic oversampling creates a balanced 50-50 training distribution.
  - **Validation**: Original test set proportions maintained for realistic performance evaluation.
  - **Result**: Models learn fraud patterns effectively while maintaining real-world applicability.

## Code Quality & Testing

### Validation Approach
- Stratified train-test splits maintain class distribution.
- Cross-validation ensures model stability.
- Fraud-focused evaluation metrics (F1, ROC-AUC, AUC-PR).
- SHAP validation with shape consistency checks.

### Error Handling & Robustness
- File existence validation before data loading.
- Missing value detection and appropriate handling strategies.
- Data type validation and conversion safeguards.
- Memory usage optimization for large datasets.

### Troubleshooting Guide
#### Common Setup Issues
1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`. Check Python version compatibility (3.11+).
2. **File Path Errors**: Verify datasets are correctly placed in `data/raw/` with matching filenames (`Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv`).
3. **Memory Issues**: Increase RAM or use data sampling if memory errors occur during processing of large datasets.
4. **Jupyter Kernel Issues**: Restart the kernel and clear outputs if cells fail to execute; ensure no stale variables persist.

#### Data Processing Issues
1. **Missing Files**: Confirm all three datasets are present and not corrupted; re-download if necessary.
2. **Encoding Issues**: Ensure CSV files are in UTF-8 encoding; convert using a text editor or `iconv` if needed.
3. **Memory Warnings**: These are normal for large datasets; processing will continue unless an error is raised.

## Dependencies
- `pandas`, `numpy` for data manipulation
- `scikit-learn` for modeling and preprocessing
- `imbalanced-learn` for SMOTE implementation
- `matplotlib`, `seaborn` for visualization
- `shap` for model explainability
- `jupyter` for the notebook environment

**Full dependency list available in `requirements.txt`**.

## Project Status
- ✅ **Task 1 - Completed**
  - Comprehensive EDA with fraud pattern identification.
  - Feature engineering with business logic justification.
  - Data preprocessing pipeline with quality assurance.
  - Class imbalance analysis and strategy development.
- ✅ **Task 2 - Completed**
  - SMOTE implementation for class balance.
  - Logistic Regression and Random Forest training.
  - Comprehensive model evaluation and comparison.
  - Performance visualization with ROC and PR curves.
  - Model selection based on fraud detection effectiveness.
- ✅ **Task 3 - Completed**
  - SHAP implementation for model interpretability.
  - Generation of Summary and Force plots.
  - Detailed analysis of global and local feature importance.
  - Insights integrated into `interim_3_report.md`.

## Contributing & Code Standards

### Development Workflow
1. Create feature branches for new development.
2. Write comprehensive inline comments explaining business logic.
3. Include docstrings for all functions.
4. Test code with sample data before committing.
5. Update documentation for any new features.

### Documentation Standards
- All functions must include docstrings with parameter descriptions.
- Complex business logic requires inline comments explaining the rationale.
- README updates are required for any structural changes.
- Report files are maintained for technical documentation.

## Repository
- **GitHub**: `https://github.com/Yihenew21/Fraud-Detection-ML`
- **Documentation**: Complete technical reports in the `reports/` directory.
- **Reproducibility**: All analysis steps are documented and executable.
- **Code Quality**: Modular design with comprehensive documentation.

## Note
- This README was last updated at 08:48 AM PDT on Wednesday, July 30, 2025, reflecting the completion of Task 3 post-deadline. Please consult the instructor regarding resubmission policies.

---
*This project demonstrates end-to-end fraud detection implementation from data preprocessing through model evaluation, selection, and explainability, with emphasis on code quality, documentation clarity, and business impact analysis.*
