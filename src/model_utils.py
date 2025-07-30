"""
Model utilities for training, evaluating, and interpreting machine learning models.

This module provides comprehensive functionality for:
- Training and hyperparameter tuning of classification models
- Model evaluation with multiple metrics
- SHAP-based model interpretability and visualization

Author: Machine Learning Team
Date: 2024
"""

import numpy as np
import time
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import shap


def train_evaluate_model(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train, tune, and evaluate Logistic Regression and Random Forest models.

    This function performs comprehensive model training including hyperparameter tuning
    using GridSearchCV, model evaluation with multiple metrics, cross-validation,
    and confusion matrix analysis for both Logistic Regression and Random Forest classifiers.

    Args:
        X_train (array-like): Training feature matrix of shape (n_samples, n_features)
        X_test (array-like): Testing feature matrix of shape (n_samples, n_features)
        y_train (array-like): Training target vector of shape (n_samples,)
        y_test (array-like): Testing target vector of shape (n_samples,)
        dataset_name (str): Name of the dataset for logging and identification purposes

    Returns:
        tuple: A 5-element tuple containing:
            - best_lr (LogisticRegression): Best tuned Logistic Regression model
            - best_rf (RandomForestClassifier): Best tuned Random Forest model
            - y_test (array-like): Original test labels
            - y_pred_lr_tuned (array-like): Logistic Regression predictions
            - y_pred_rf_tuned (array-like): Random Forest predictions

    Example:
        >>> best_lr, best_rf, y_test, y_pred_lr, y_pred_rf = train_evaluate_model(
        ...     X_train, X_test, y_train, y_test, "Heart Disease"
        ... )
    """
    # Hyperparameter tuning with GridSearchCV, using all CPU cores for parallel processing
    # Define parameter grid for Logistic Regression - testing regularization strength
    lr_param_grid = {"C": [1, 10]}  # C values: lower = more regularization

    # Create GridSearchCV object for Logistic Regression
    lr_grid = GridSearchCV(
        LogisticRegression(
            random_state=42, max_iter=1000
        ),  # Base estimator with fixed seed
        lr_param_grid,  # Parameter combinations to test
        cv=5,  # 5-fold cross-validation
        scoring="f1",  # Optimize for F1-score (good for imbalanced data)
        n_jobs=-1,  # Use all available CPU cores
    )

    # Fit the grid search to find best hyperparameters
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_  # Extract the best model

    # Define parameter grid for Random Forest - testing ensemble size and tree depth
    rf_param_grid = {
        "n_estimators": [50],
        "max_depth": [20],
    }  # Limited grid for efficiency

    # Create GridSearchCV object for Random Forest
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),  # Base estimator with fixed seed
        rf_param_grid,  # Parameter combinations to test
        cv=5,  # 5-fold cross-validation
        scoring="f1",  # Optimize for F1-score
        n_jobs=-1,  # Use all available CPU cores
    )

    # Fit the grid search to find best hyperparameters
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_  # Extract the best model

    # Generate predictions using the tuned models
    y_pred_lr_tuned = best_lr.predict(X_test)  # Logistic Regression predictions
    y_pred_rf_tuned = best_rf.predict(X_test)  # Random Forest predictions

    # Calculate comprehensive evaluation metrics for Logistic Regression
    metrics_lr_tuned = {
        "accuracy": accuracy_score(y_test, y_pred_lr_tuned),  # Overall correctness
        "precision": precision_score(
            y_test, y_pred_lr_tuned
        ),  # True positives / (True + False positives)
        "recall": recall_score(
            y_test, y_pred_lr_tuned
        ),  # True positives / (True positives + False negatives)
        "f1": f1_score(
            y_test, y_pred_lr_tuned
        ),  # Harmonic mean of precision and recall
        "roc_auc": roc_auc_score(y_test, y_pred_lr_tuned),  # Area under ROC curve
    }

    # Calculate comprehensive evaluation metrics for Random Forest
    metrics_rf_tuned = {
        "accuracy": accuracy_score(y_test, y_pred_rf_tuned),  # Overall correctness
        "precision": precision_score(
            y_test, y_pred_rf_tuned
        ),  # True positives / (True + False positives)
        "recall": recall_score(
            y_test, y_pred_rf_tuned
        ),  # True positives / (True positives + False negatives)
        "f1": f1_score(
            y_test, y_pred_rf_tuned
        ),  # Harmonic mean of precision and recall
        "roc_auc": roc_auc_score(y_test, y_pred_rf_tuned),  # Area under ROC curve
    }

    # Print evaluation results for comparison
    print(f"Tuned {dataset_name} Logistic Regression Metrics:", metrics_lr_tuned)
    print(f"Best C:", lr_grid.best_params_["C"])  # Best regularization parameter
    print(f"Tuned {dataset_name} Random Forest Metrics:", metrics_rf_tuned)
    print(f"Best Parameters:", rf_grid.best_params_)  # Best hyperparameters

    # Perform cross-validation to assess model stability and generalization
    lr_cv_scores = cross_val_score(
        best_lr, X_train, y_train, cv=5, scoring="f1"
    )  # 5-fold CV for LR
    rf_cv_scores = cross_val_score(
        best_rf, X_train, y_train, cv=5, scoring="f1"
    )  # 5-fold CV for RF

    # Print cross-validation results
    print(f"{dataset_name} Logistic Regression CV F1 Scores:", lr_cv_scores)
    print(f"Mean CV F1 Score:", lr_cv_scores.mean())  # Average performance across folds
    print(f"{dataset_name} Random Forest CV F1 Scores:", rf_cv_scores)
    print(f"Mean CV F1 Score:", rf_cv_scores.mean())  # Average performance across folds

    # Generate confusion matrices for detailed error analysis
    cm_lr = confusion_matrix(y_test, y_pred_lr_tuned)  # LR confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf_tuned)  # RF confusion matrix

    # Print confusion matrices for manual inspection
    print(f"{dataset_name} Logistic Regression Confusion Matrix:\\n", cm_lr)
    print(f"{dataset_name} Random Forest Confusion Matrix:\\n", cm_rf)

    # Return all important objects for further analysis
    return best_lr, best_rf, y_test, y_pred_lr_tuned, y_pred_rf_tuned


def compute_shap_values(model, X_test, dataset_name, n_samples=1000):
    """
    Compute SHAP values for the given model on a subset of test data.

    SHAP (SHapley Additive exPlanations) values provide a unified measure of feature
    importance that explains individual predictions by quantifying the contribution
    of each feature to the prediction difference from the baseline.

    Args:
        model (sklearn estimator): Trained tree-based model (e.g., Random Forest)
                                 Must be compatible with shap.TreeExplainer
        X_test (array-like): Test feature matrix of shape (n_samples, n_features)
        dataset_name (str): Name of the dataset for logging and identification
        n_samples (int, optional): Maximum number of samples to use for SHAP computation.
                                 Defaults to 1000 for computational efficiency

    Returns:
        tuple: A 2-element tuple containing:
            - shap_values (array-like): SHAP values for each sample and feature
            - X_test_subset (array-like): Subset of test data used for computation

    Note:
        Uses random sampling if X_test contains more than n_samples rows to
        maintain computational efficiency while preserving representativeness.

    Example:
        >>> shap_vals, X_subset = compute_shap_values(rf_model, X_test, "Heart Disease")
    """
    # Determine optimal subset size for SHAP computation (balance accuracy vs efficiency)
    if len(X_test) > n_samples:
        # Randomly sample without replacement to maintain data distribution
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_test_subset = X_test[idx]  # Create subset for SHAP analysis
    else:
        # Use all data if dataset is small enough
        X_test_subset = X_test

    # Initialize SHAP TreeExplainer for tree-based models (RF, XGBoost, etc.)
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values - this may take time for large datasets
    shap_values = explainer.shap_values(X_test_subset)

    # Log completion for monitoring progress
    print(f"SHAP values computed for {dataset_name} with {n_samples} samples")

    return shap_values, X_test_subset


def plot_shap_summary(shap_values, X_test, dataset_name, feature_names=None):
    """
    Generate and save a SHAP summary plot for model interpretability.

    Creates a comprehensive summary plot showing the impact of each feature
    on model predictions. Features are ranked by importance, and the plot
    shows the distribution of SHAP values across all samples.

    Args:
        shap_values (array-like): SHAP values from compute_shap_values function
                                Shape should be (n_samples, n_features)
        X_test (array-like): Subset of test data corresponding to SHAP values
                           Shape should be (n_samples, n_features)
        dataset_name (str): Name of the dataset for plot title and filename
        feature_names (list, optional): List of feature names for plot labels.
                                      If None, uses default feature indices

    Returns:
        None: Function saves plot to file and closes matplotlib figure

    Note:
        - Plot is saved to 'plots/' directory with standardized filename
        - Uses class 1 SHAP values for binary classification (positive class)
        - Automatically handles figure sizing and layout optimization

    Example:
        >>> plot_shap_summary(shap_vals, X_subset, "Heart Disease", feature_names)
    """
    # Create figure with appropriate size for readability
    plt.figure(figsize=(10, 6))  # Width=10, Height=6 inches

    # Generate SHAP summary plot without displaying (save only)
    # Uses class 1 (positive class) for binary classification
    shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)

    # Add descriptive title with dataset name
    plt.title(f"{dataset_name} SHAP Summary Plot")

    # Optimize layout to prevent label cutoff
    plt.tight_layout()

    # Save plot with standardized naming convention
    plt.savefig(f'plots/{dataset_name.lower().replace(" ", "_")}_shap_summary.png')

    # Close figure to free memory
    plt.close()


def plot_shap_force(shap_values, X_test, dataset_name, sample_idx=0):
    """
    Generate and save a SHAP force plot for a single sample prediction explanation.

    Creates a force plot that shows how each feature contributes to pushing
    the model output from the base value (average prediction) to the final
    prediction for a specific sample.

    Args:
        shap_values (array-like): SHAP values from compute_shap_values function
                                Must contain explainer information for base values
        X_test (array-like): Subset of test data corresponding to SHAP values
                           Shape should be (n_samples, n_features)
        dataset_name (str): Name of the dataset for plot title and filename
        sample_idx (int, optional): Index of the sample to explain (0-based).
                                  Defaults to 0 (first sample)

    Returns:
        None: Function saves plot to file and closes matplotlib figure

    Note:
        - Requires shap_values to contain TreeExplainer information
        - Uses class 1 (positive class) for binary classification
        - Plot shows features pushing prediction higher (red) or lower (blue)
        - Saves to 'plots/' directory with sample index in filename

    Example:
        >>> plot_shap_force(shap_vals, X_subset, "Heart Disease", sample_idx=5)
    """
    # Create TreeExplainer from SHAP values (assumes TreeExplainer was used)
    explainer = shap.TreeExplainer(shap_values)

    # Generate force plot for specified sample
    # Uses expected_value[1] for positive class baseline
    shap.force_plot(
        explainer.expected_value[1],  # Base value (average prediction for class 1)
        shap_values[1][sample_idx],  # SHAP values for the specific sample
        X_test[sample_idx],  # Feature values for the specific sample
        matplotlib=True,  # Use matplotlib backend for saving
        show=False,  # Don't display, only save
    )

    # Add descriptive title with dataset and sample information
    plt.title(f"{dataset_name} SHAP Force Plot - Sample {sample_idx}")

    # Save plot with standardized naming including sample index
    plt.savefig(
        f'plots/{dataset_name.lower().replace(" ", "_")}_shap_force_{sample_idx}.png'
    )

    # Close figure to free memory
    plt.close()
