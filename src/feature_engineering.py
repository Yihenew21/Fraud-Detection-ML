# src/feature_engineering.py
import pandas as pd


def add_time_features(df):
    """
    Add time-based features to the DataFrame:
        - time_since_signup: Hours between signup and purchase.
        - hour_of_day: Hour of purchase (0-23).
        - day_of_week: Day of week of purchase (0=Monday).

    Parameters:
        df (pd.DataFrame): DataFrame with 'signup_time' and 'purchase_time' columns (as datetime).

    Returns:
        pd.DataFrame: DataFrame with new time-based features.
    """
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
