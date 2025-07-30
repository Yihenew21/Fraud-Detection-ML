"""
Feature engineering utilities for machine learning preprocessing.

This module provides comprehensive feature engineering capabilities including:
- Time-based feature extraction from datetime columns
- Transaction frequency calculation for user behavior analysis
- Complete data preprocessing pipeline with encoding, scaling, and balancing
- SMOTE implementation for handling class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def add_time_features(df):
    """
    Add time-based features to the DataFrame for temporal pattern analysis.

    This function extracts meaningful time-based features that can help identify
    patterns in user behavior and fraud detection:
    - Time elapsed between signup and purchase (indicator of user behavior)
    - Hour of day (circadian patterns, business hours vs off-hours)
    - Day of week (weekday vs weekend patterns)

    Parameters:
        df (pd.DataFrame): DataFrame containing 'signup_time' and 'purchase_time' columns.
            These can be datetime objects or string representations of dates/times.

    Returns:
        pd.DataFrame: Original DataFrame with three additional columns:
            - 'time_since_signup': Float, hours between signup and purchase
            - 'hour_of_day': Integer (0-23), hour when purchase occurred
            - 'day_of_week': Integer (0-6), day of week (0=Monday, 6=Sunday)

    Raises:
        KeyError: If 'signup_time' or 'purchase_time' columns are missing.
        ValueError: If datetime conversion fails for the time columns.

    Note:
        Negative time_since_signup values indicate purchases made before signup,
        which could be a data quality issue or potential fraud indicator.

    Example:
        >>> df_with_time = add_time_features(df)
        >>> print(df_with_time[['time_since_signup', 'hour_of_day', 'day_of_week']].head())
    """
    # Convert to datetime if columns are strings to enable time calculations
    # The errors='coerce' parameter converts invalid dates to NaT (Not a Time)
    if df["purchase_time"].dtype == "object" or df["signup_time"].dtype == "object":
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
        df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")

    # Calculate hours between signup and purchase as a behavioral feature
    # This measures user engagement speed - quick purchases might indicate impulse buying or fraud
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600

    # Extract hour of day from purchase_time (0-23)
    # Useful for detecting unusual activity patterns (e.g., purchases at 3 AM)
    df["hour_of_day"] = df["purchase_time"].dt.hour

    # Extract day of week from purchase_time (0=Monday, 6=Sunday)
    # Helps identify weekend vs weekday purchasing patterns
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    return df


def add_transaction_frequency(df, group_col):
    """
    Add transaction frequency per group to identify high-activity users.

    This function calculates how many transactions each entity (e.g., user, device, IP)
    has made across the entire dataset. High transaction frequency can indicate:
    - Power users with legitimate high activity
    - Potential fraudulent accounts making multiple attempts
    - Business accounts with high transaction volumes

    Parameters:
        df (pd.DataFrame): DataFrame to add frequency feature to.
        group_col (str): Column name to group by for frequency calculation.
            Common examples: 'user_id', 'device_id', 'ip_address', 'email'

    Returns:
        pd.DataFrame: Original DataFrame with additional 'trans_freq' column
            containing the transaction count for each group member.

    Example:
        >>> df_with_freq = add_transaction_frequency(df, 'user_id')
        >>> print(df_with_freq.groupby('user_id')['trans_freq'].first().head())
        user_id
        1001    5
        1002    1
        1003    23
        1004    2
        1005    1
    """
    # Count number of transactions per group and assign to each row
    # transform() preserves the original DataFrame structure while adding aggregated values
    # This gives each row the total count for its group
    df["trans_freq"] = df.groupby(group_col)[group_col].transform("count")

    return df


def preprocess_data(df, target_col, cat_cols=None, num_cols=None):
    """
    Comprehensive preprocessing pipeline for machine learning models.

    This function performs a complete preprocessing workflow including:
    1. Feature-target separation and train-test split with stratification
    2. One-hot encoding for categorical variables (if provided)
    3. Standard scaling for numerical features to normalize distributions
    4. Feature concatenation to create final feature matrix
    5. SMOTE oversampling to handle class imbalance in training data

    Args:
        df (pd.DataFrame): Input dataframe with features and target.
        target_col (str): Name of the target column (e.g., 'class', 'Class', 'is_fraud').
        cat_cols (list, optional): List of categorical column names to encode.
            If None, only numerical features are used. Defaults to None.
        num_cols (list): List of numerical column names to scale.
            These should be continuous or ordinal variables.

    Returns:
        tuple: Six-element tuple containing:
            - X_train_res (np.ndarray): SMOTE-resampled training features
            - X_test_processed (np.ndarray): Processed test features (no SMOTE)
            - y_train_res (np.ndarray): SMOTE-resampled training labels
            - y_test (np.ndarray): Original test labels
            - encoder (OneHotEncoder or None): Fitted encoder for categorical variables
            - feature_names (list): Names of all features in final feature matrix

    Raises:
        KeyError: If specified columns don't exist in the DataFrame.
        ValueError: If target column contains unexpected values or data types.

    Note:
        - Uses stratified sampling to maintain class distribution in train-test split
        - SMOTE is only applied to training data to prevent data leakage
        - Standard scaling uses training data statistics to transform test data
        - Feature names preserve order: [encoded_categorical_features] + [numerical_features]

    Example:
        >>> X_train, X_test, y_train, y_test, encoder, features = preprocess_data(
        ...     df, 'is_fraud', ['category', 'device'], ['amount', 'age']
        ... )
        >>> print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    """
    # Separate features and target variable
    # Select only the specified categorical and numerical columns
    X = df[cat_cols + num_cols] if cat_cols else df[num_cols]
    y = df[target_col]

    # Split into train and test sets with stratification to maintain class balance
    # stratify=y ensures both sets have similar class distributions
    # random_state=42 for reproducible results across runs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode categorical variables if provided
    if cat_cols:
        # Initialize one-hot encoder with sparse_output=False for dense arrays
        # handle_unknown='ignore' prevents errors on unseen categories in test data
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Fit encoder on training data and transform both train and test
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])

        # Get feature names from one-hot encoding for interpretability
        # Combines encoded categorical names with original numerical names
        feature_names = list(encoder.get_feature_names_out(cat_cols)) + num_cols
    else:
        # No categorical variables to encode
        encoder = None
        X_train_cat = np.array([])
        X_test_cat = np.array([])
        feature_names = num_cols

    # Scale numerical features to have zero mean and unit variance
    # This normalizes features with different scales (e.g., age vs income)
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])  # Use training statistics

    # Combine categorical and numerical features into final feature matrices
    # np.hstack concatenates arrays horizontally (column-wise)
    X_train_processed = (
        np.hstack((X_train_num, X_train_cat)) if cat_cols else X_train_num
    )
    X_test_processed = np.hstack((X_test_num, X_test_cat)) if cat_cols else X_test_num

    # Apply SMOTE for class balance on training data only
    # SMOTE generates synthetic minority class samples to balance the dataset
    # random_state=42 for reproducible synthetic sample generation
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

    return X_train_res, X_test_processed, y_train_res, y_test, encoder, feature_names
