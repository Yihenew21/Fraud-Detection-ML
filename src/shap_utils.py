"""
SHAP utilities for model interpretability and explainability.

This module provides specialized functionality for computing and visualizing
SHAP (SHapley Additive exPlanations) values, which help explain individual
predictions by quantifying feature contributions.

Key Features:
- Robust SHAP value computation with automatic data type handling
- Flexible handling of different SHAP value structures (3D arrays vs lists)
- Summary and force plot generation for comprehensive model interpretation
- Efficient sampling for large datasets to maintain computational feasibility

Author: ML Interpretability Team
Date: 2024
"""

import shap
import numpy as np
import matplotlib.pyplot as plt


def compute_shap_values(model, X_test, dataset_name):
    """
    Compute SHAP values for the given model and test data with robust error handling.

    This function computes SHAP values using TreeExplainer and handles various
    SHAP value output formats automatically. It includes efficient sampling for
    large datasets and comprehensive logging for debugging purposes.

    Args:
        model (sklearn estimator): Trained tree-based model (e.g., RandomForestClassifier)
                                 Must be compatible with shap.TreeExplainer
        X_test (array-like): Test data features of shape (n_samples, n_features)
                           Can be pandas DataFrame, numpy array, or similar
        dataset_name (str): Name of the dataset for logging and identification purposes

    Returns:
        tuple: A 3-element tuple containing:
            - shap_values_class1 (np.ndarray): SHAP values for positive class
                                             Shape: (n_samples, n_features)
            - X_test_subset (np.ndarray): Subset of test data used for computation
                                        Shape: (n_samples, n_features)
            - explainer (shap.TreeExplainer): Fitted SHAP explainer object
                                            for additional analysis

    Raises:
        ValueError: If SHAP values structure is unexpected or incompatible

    Note:
        - Automatically samples up to 1000 rows for computational efficiency
        - Handles both 3D array and list formats of SHAP values
        - Always extracts class 1 (positive class) for binary classification

    Example:
        >>> shap_vals, X_subset, explainer = compute_shap_values(
        ...     rf_model, X_test, "Heart Disease Dataset"
        ... )
    """
    # Ensure input is numpy array for consistent indexing and operations
    X_test = np.asarray(X_test)  # Convert pandas DataFrame or list to numpy array

    # Use efficient subset of data for SHAP (up to 1000 samples for speed)
    n_samples = min(1000, X_test.shape[0])  # Choose smaller of 1000 or dataset size

    # Generate random indices without replacement to maintain data distribution
    indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
    X_test_subset = X_test[indices]  # Create representative subset

    # Initialize TreeExplainer for tree-based models (handles RF, XGBoost, etc.)
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values - this is the computationally expensive step
    shap_values = explainer.shap_values(X_test_subset)

    # Handle different SHAP value output formats robustly
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Format: 3D array (n_samples, n_features, n_classes)
        # Extract class 1 (positive class) predictions
        shap_values_class1 = shap_values[:, :, 1]  # Shape: (n_samples, n_features)
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        # Format: List of two 2D arrays [class_0_shap, class_1_shap]
        # Use class 1 (positive class) SHAP values
        shap_values_class1 = shap_values[1]  # Shape: (n_samples, n_features)
    else:
        # Unexpected format - raise informative error for debugging
        raise ValueError(
            f"Unexpected SHAP values structure for {dataset_name}: {shap_values}"
        )

    # Log completion and shape information for verification
    print(f"SHAP values computed for {dataset_name} with {n_samples} samples")
    print(f"SHAP values shape for {dataset_name}: {shap_values_class1.shape}")
    print(f"X_test_subset shape for {dataset_name}: {X_test_subset.shape}")

    return shap_values_class1, X_test_subset, explainer


def plot_shap_summary(shap_values, X_test, dataset_name, feature_names):
    """
    Generate and save a comprehensive SHAP summary plot for feature importance analysis.

    Creates a detailed summary plot that visualizes the impact of each feature
    on model predictions. Features are automatically ranked by importance,
    and the plot shows both the magnitude and direction of feature effects
    across all samples in the dataset.

    Args:
        shap_values (np.ndarray): SHAP values for positive class from compute_shap_values
                                Shape should be (n_samples, n_features)
        X_test (np.ndarray): Test data subset corresponding to SHAP values
                           Shape should be (n_samples, n_features)
        dataset_name (str): Name of the dataset for plot title and filename generation
        feature_names (list): List of human-readable feature names for plot labels
                            Length should match number of features in X_test

    Returns:
        None: Function saves plot to '../plots/' directory and closes figure

    Note:
        - Features are automatically sorted by average absolute SHAP value
        - Color coding shows feature value (red=high, blue=low)
        - Dot position shows SHAP value impact (right=positive, left=negative)
        - Plot saved with standardized filename format

    Example:
        >>> plot_shap_summary(
        ...     shap_vals, X_subset, "Heart Disease",
        ...     ["age", "cholesterol", "blood_pressure"]
        ... )
    """
    # Create figure with optimal dimensions for feature visualization
    plt.figure(figsize=(10, 6))  # Wide format for better feature name readability

    # Generate SHAP summary plot with custom styling
    shap.summary_plot(
        shap_values,  # SHAP values for positive class
        X_test,  # Corresponding feature values
        feature_names=feature_names,  # Human-readable feature labels
        show=False,  # Don't display interactively, only save
    )

    # Add informative title with dataset identification
    plt.title(f"{dataset_name} SHAP Summary Plot")

    # Optimize layout to prevent label truncation
    plt.tight_layout()

    # Save to plots directory with standardized naming convention
    plt.savefig(f'../plots/{dataset_name.lower().replace(" ", "_")}_shap_summary.png')

    # Close figure to free memory and prevent display issues
    plt.close()


def plot_shap_force(shap_values, X_test, dataset_name, explainer):
    """
    Generate and save a SHAP force plot for detailed single-sample prediction explanation.

    Creates an interactive-style force plot that shows exactly how each feature
    contributes to pushing the model's prediction away from the baseline (expected value)
    toward the final prediction for a specific sample.

    Args:
        shap_values (np.ndarray): SHAP values for positive class
                                Shape: (n_samples, n_features)
        X_test (np.ndarray): Test data subset corresponding to SHAP values
                           Shape: (n_samples, n_features)
        dataset_name (str): Name of the dataset for plot title and filename
        explainer (shap.TreeExplainer): Fitted SHAP explainer object containing
                                      baseline values and model information

    Returns:
        None: Function saves plot to '../plots/' directory and prints confirmation

    Note:
        - Uses the first sample (index 0) for explanation by default
        - Red bars show features pushing prediction higher than baseline
        - Blue bars show features pushing prediction lower than baseline
        - Bar length represents magnitude of feature contribution
        - Baseline value comes from explainer.expected_value[1] (positive class)

    Example:
        >>> plot_shap_force(shap_vals, X_subset, "Heart Disease", explainer)
        Force plot saved for Heart Disease
    """
    # Extract SHAP values and features for the first sample (index 0)
    shap_values_sample = shap_values[0]  # 1D array: SHAP values for each feature
    X_test_sample = X_test[0]  # 1D array: feature values for the sample

    # Create new figure for the force plot
    plt.figure()

    # Generate force plot using matplotlib backend for file saving
    shap.force_plot(
        explainer.expected_value[1],  # Baseline prediction (average for positive class)
        shap_values_sample,  # SHAP contributions for each feature
        X_test_sample,  # Actual feature values for context
        matplotlib=True,  # Use matplotlib instead of JavaScript for saving
        show=False,  # Don't display interactively
    )

    # Add descriptive title with dataset and sample information
    plt.title(f"{dataset_name} SHAP Force Plot for Sample 0")

    # Save plot with standardized naming convention including sample identifier
    plt.savefig(f'../plots/{dataset_name.lower().replace(" ", "_")}_shap_force_0.png')

    # Close figure to prevent memory leaks
    plt.close()

    # Provide confirmation that plot was saved successfully
    print(f"Force plot saved for {dataset_name}")
