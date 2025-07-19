# src/feature_engineering.py
import pandas as pd


def add_time_features(df):
    """Add time-based features: time_since_signup, hour_of_day, day_of_week."""
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    return df


def add_transaction_frequency(df, group_col):
    """Add transaction frequency per group (e.g., user_id) across the dataset."""
    df["trans_freq"] = df.groupby(group_col)[group_col].transform("count")
    return df
