import shap
import numpy as np
import matplotlib.pyplot as plt


def compute_shap_values(model, X_test, dataset_name):
    """
    Compute SHAP values for the given model and test data.

    Args:
        model: Trained model (e.g., RandomForestClassifier).
        X_test (np.ndarray): Test data features.
        dataset_name (str): Name of the dataset for logging.

    Returns:
        tuple: (shap_values, X_test_subset, explainer)
    """
    # Ensure X_test is a numpy array
    X_test = np.asarray(X_test)
    # Use a subset of data for SHAP (up to 1000 samples for efficiency)
    n_samples = min(1000, X_test.shape[0])
    indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
    X_test_subset = X_test[indices]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_subset)

    # Handle SHAP values structure
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array (n_samples, n_features, n_classes), extract class 1
        shap_values_class1 = shap_values[:, :, 1]  # Shape: (n_samples, n_features)
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        # List of two 2D arrays, use class 1
        shap_values_class1 = shap_values[1]
    else:
        raise ValueError(
            f"Unexpected SHAP values structure for {dataset_name}: {shap_values}"
        )

    print(f"SHAP values computed for {dataset_name} with {n_samples} samples")
    print(f"SHAP values shape for {dataset_name}: {shap_values_class1.shape}")
    print(f"X_test_subset shape for {dataset_name}: {X_test_subset.shape}")
    return shap_values_class1, X_test_subset, explainer


def plot_shap_summary(shap_values, X_test, dataset_name, feature_names):
    """
    Generate and save a SHAP summary plot.

    Args:
        shap_values: SHAP values computed for the model (for class 1).
        X_test (np.ndarray): Test data subset.
        dataset_name (str): Name of the dataset.
        feature_names (list): List of feature names (optional).
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"{dataset_name} SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f'../plots/{dataset_name.lower().replace(" ", "_")}_shap_summary.png')
    plt.close()


def plot_shap_force(shap_values, X_test, dataset_name, explainer):
    """
    Generate and save a SHAP force plot for the first sample.

    Args:
        shap_values: SHAP values computed for the model.
        X_test (np.ndarray): Test data subset.
        dataset_name (str): Name of the dataset.
        explainer: SHAP explainer object.
    """
    # Use the first sample's SHAP values and features
    shap_values_sample = shap_values[0]  # 1D array for the first sample
    X_test_sample = X_test[0]  # 1D array for the first sample

    # Generate force plot with base value from explainer
    plt.figure()
    shap.force_plot(
        explainer.expected_value[1],
        shap_values_sample,
        X_test_sample,
        matplotlib=True,
        show=False,
    )
    plt.title(f"{dataset_name} SHAP Force Plot for Sample 0")
    plt.savefig(f'../plots/{dataset_name.lower().replace(" ", "_")}_shap_force_0.png')
    plt.close()
    print(f"Force plot saved for {dataset_name}")
