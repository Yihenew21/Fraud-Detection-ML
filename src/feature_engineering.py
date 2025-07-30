import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def add_time_features(df):
    """
    Add time-based features to the DataFrame:
        - time_since_signup: Hours between signup and purchase.
        - hour_of_day: Hour of purchase (0-23).
        - day_of_week: Day of week of purchase (0=Monday).

    Parameters:
        df (pd.DataFrame): DataFrame with 'signup_time' and 'purchase_time' columns (as datetime or string).

    Returns:
        pd.DataFrame: DataFrame with new time-based features.
    """
    # Convert to datetime if columns are strings
    if df["purchase_time"].dtype == "object" or df["signup_time"].dtype == "object":
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
        df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")

    # Calculate hours between signup and purchase
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600
    # Extract hour of day from purchase_time
    df["hour_of_day"] = df["purchase_time"].dt.hour
    # Extract day of week from purchase_time (0=Monday, 6=Sunday)
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    return df


def add_transaction_frequency(df, group_col):
    """
    Add transaction frequency per group (e.g., user_id) across the dataset.

    Parameters:
        df (pd.DataFrame): DataFrame to add frequency feature to.
        group_col (str): Column to group by (e.g., 'user_id').

    Returns:
        pd.DataFrame: DataFrame with new 'trans_freq' column.
    """
    # Count number of transactions per group and assign to each row
    df["trans_freq"] = df.groupby(group_col)[group_col].transform("count")
    return df


def preprocess_data(df, target_col, cat_cols=None, num_cols=None):
    """
    Preprocess dataset by encoding categoricals, scaling numericals, and applying SMOTE.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Target column name ('class' or 'Class').
        cat_cols (list): List of categorical column names (default None for credit card).
        num_cols (list): List of numerical column names.

    Returns:
        tuple: (X_train_res, X_test_processed, y_train_res, y_test, encoder, feature_names)
    """
    # Separate features and target
    X = df[cat_cols + num_cols] if cat_cols else df[num_cols]
    y = df[target_col]

    # Split into train and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode categorical variables if provided
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])
        # Get feature names from one-hot encoding
        feature_names = list(encoder.get_feature_names_out(cat_cols)) + num_cols
    else:
        encoder = None
        X_train_cat = np.array([])
        X_test_cat = np.array([])
        feature_names = num_cols

    # Scale numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    # Combine features
    X_train_processed = (
        np.hstack((X_train_num, X_train_cat)) if cat_cols else X_train_num
    )
    X_test_processed = np.hstack((X_test_num, X_test_cat)) if cat_cols else X_test_num

    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

    return X_train_res, X_test_processed, y_train_res, y_test, encoder, feature_names
