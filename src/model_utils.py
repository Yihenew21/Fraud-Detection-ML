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

    Args:
        X_train, X_test, y_train, y_test: Training and test data.
        dataset_name (str): Name of the dataset for logging.

    Returns:
        tuple: (best_lr, best_rf, y_test, y_pred_lr, y_pred_rf)
    """
    # Hyperparameter tuning with GridSearchCV, using all CPU cores
    lr_param_grid = {"C": [1, 10]}
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        lr_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_

    rf_param_grid = {"n_estimators": [50], "max_depth": [20]}
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # Tuned predictions and metrics
    y_pred_lr_tuned = best_lr.predict(X_test)
    y_pred_rf_tuned = best_rf.predict(X_test)

    metrics_lr_tuned = {
        "accuracy": accuracy_score(y_test, y_pred_lr_tuned),
        "precision": precision_score(y_test, y_pred_lr_tuned),
        "recall": recall_score(y_test, y_pred_lr_tuned),
        "f1": f1_score(y_test, y_pred_lr_tuned),
        "roc_auc": roc_auc_score(y_test, y_pred_lr_tuned),
    }
    metrics_rf_tuned = {
        "accuracy": accuracy_score(y_test, y_pred_rf_tuned),
        "precision": precision_score(y_test, y_pred_rf_tuned),
        "recall": recall_score(y_test, y_pred_rf_tuned),
        "f1": f1_score(y_test, y_pred_rf_tuned),
        "roc_auc": roc_auc_score(y_test, y_pred_rf_tuned),
    }

    print(f"Tuned {dataset_name} Logistic Regression Metrics:", metrics_lr_tuned)
    print(f"Best C:", lr_grid.best_params_["C"])
    print(f"Tuned {dataset_name} Random Forest Metrics:", metrics_rf_tuned)
    print(f"Best Parameters:", rf_grid.best_params_)

    # Cross-validation
    lr_cv_scores = cross_val_score(best_lr, X_train, y_train, cv=5, scoring="f1")
    rf_cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring="f1")
    print(f"{dataset_name} Logistic Regression CV F1 Scores:", lr_cv_scores)
    print(f"Mean CV F1 Score:", lr_cv_scores.mean())
    print(f"{dataset_name} Random Forest CV F1 Scores:", rf_cv_scores)
    print(f"Mean CV F1 Score:", rf_cv_scores.mean())

    # Confusion Matrix
    cm_lr = confusion_matrix(y_test, y_pred_lr_tuned)
    cm_rf = confusion_matrix(y_test, y_pred_rf_tuned)
    print(f"{dataset_name} Logistic Regression Confusion Matrix:\\n", cm_lr)
    print(f"{dataset_name} Random Forest Confusion Matrix:\\n", cm_rf)

    return best_lr, best_rf, y_test, y_pred_lr_tuned, y_pred_rf_tuned


def compute_shap_values(model, X_test, dataset_name, n_samples=1000):
    """
    Compute SHAP values for the given model on a subset of test data.

    Args:
        model: Trained model (e.g., Random Forest).
        X_test: Test feature matrix.
        dataset_name (str): Name of the dataset for logging.
        n_samples (int): Number of samples to use for SHAP (default 1000).

    Returns:
        shap_values: SHAP values for interpretation.
    """
    # Use a subset for efficiency
    if len(X_test) > n_samples:
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_test_subset = X_test[idx]
    else:
        X_test_subset = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_subset)
    print(f"SHAP values computed for {dataset_name} with {n_samples} samples")
    return shap_values, X_test_subset


def plot_shap_summary(shap_values, X_test, dataset_name, feature_names=None):
    """
    Generate and save a SHAP summary plot.

    Args:
        shap_values: SHAP values from compute_shap_values.
        X_test: Subset of test data.
        dataset_name (str): Name of the dataset.
        feature_names (list): List of feature names (optional).
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    plt.title(f"{dataset_name} SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_name.lower().replace(" ", "_")}_shap_summary.png')
    plt.close()


def plot_shap_force(shap_values, X_test, dataset_name, sample_idx=0):
    """
    Generate and save a SHAP force plot for a single sample.

    Args:
        shap_values: SHAP values from compute_shap_values.
        X_test: Subset of test data.
        dataset_name (str): Name of the dataset.
        sample_idx (int): Index of the sample to plot (default 0).
    """
    explainer = shap.TreeExplainer(shap_values)
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][sample_idx],
        X_test[sample_idx],
        matplotlib=True,
        show=False,
    )
    plt.title(f"{dataset_name} SHAP Force Plot - Sample {sample_idx}")
    plt.savefig(
        f'plots/{dataset_name.lower().replace(" ", "_")}_shap_force_{sample_idx}.png'
    )
    plt.close()
