# src/data_utils.py
import pandas as pd


def load_data(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def clean_data(df, datetime_cols=None, fillna_cols=None):
    """Clean DataFrame by removing duplicates and handling missing values."""
    df = df.drop_duplicates()
    if datetime_cols:
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col])
    if fillna_cols:
        for col, strategy in fillna_cols.items():
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def merge_ip_to_country(df, ip_df):
    """Merge e-commerce DataFrame with IP-to-country mapping."""
    df["ip_address"] = df["ip_address"].astype(int)

    def map_ip(ip):
        match = ip_df[
            (ip_df["lower_bound_ip_address"] <= ip)
            & (ip_df["upper_bound_ip_address"] >= ip)
        ]
        return match["country"].iloc[0] if not match.empty else "Unknown"

    df["country"] = df["ip_address"].apply(map_ip)
    return df
