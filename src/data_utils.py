# src/data_utils.py
import pandas as pd


def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)


def clean_data(df, datetime_cols=None, fillna_cols=None):
    """
    Clean a DataFrame by removing duplicates, converting columns to datetime, and imputing missing values.

    Parameters:
        df (pd.DataFrame): Input DataFrame to clean.
        datetime_cols (list of str, optional): Columns to convert to datetime.
        fillna_cols (dict, optional): Dictionary mapping column names to fill strategies ('median' or 'mode').

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.drop_duplicates()  # Remove duplicate rows to ensure data integrity
    if datetime_cols:
        for col in datetime_cols:
            # Convert specified columns to datetime format
            df[col] = pd.to_datetime(df[col])
    if fillna_cols:
        for col, strategy in fillna_cols.items():
            if strategy == "median":
                # Fill missing values with median for numerical columns
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                # Fill missing values with mode for categorical columns
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def merge_ip_to_country(df, ip_df):
    """
    Merge an e-commerce DataFrame with an IP-to-country mapping DataFrame.
    For each IP address in the e-commerce data, find the corresponding country based on IP range.

    Parameters:
        df (pd.DataFrame): E-commerce DataFrame with an 'ip_address' column (as int).
        ip_df (pd.DataFrame): DataFrame with 'lower_bound_ip_address', 'upper_bound_ip_address', and 'country'.

    Returns:
        pd.DataFrame: DataFrame with an added 'country' column.
    """
    df["ip_address"] = df["ip_address"].astype(int)  # Ensure IP addresses are integers

    def map_ip(ip):
        # Find the country for a given IP by checking which range it falls into
        match = ip_df[
            (ip_df["lower_bound_ip_address"] <= ip)
            & (ip_df["upper_bound_ip_address"] >= ip)
        ]
        # Return the country if found, else 'Unknown'
        return match["country"].iloc[0] if not match.empty else "Unknown"

    # Apply the mapping function to each IP address
    df["country"] = df["ip_address"].apply(map_ip)
    return df
