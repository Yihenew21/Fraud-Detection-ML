# src/data_utils.py
"""
Data utility functions for loading, cleaning, and processing datasets.

This module provides essential data preprocessing capabilities including:
- CSV file loading with pandas
- Data cleaning operations (duplicates, datetime conversion, missing value imputation)
- IP address to country mapping for e-commerce data
"""

import pandas as pd


def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    This function provides a simple wrapper around pandas.read_csv() for
    consistent data loading across the project.

    Parameters:
        file_path (str): Path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame with all original columns and rows.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file is malformed.

    Example:
        >>> df = load_data('data/transactions.csv')
        >>> print(df.shape)
        (1000, 15)
    """
    # Load CSV file using pandas with default parameters
    return pd.read_csv(file_path)


def clean_data(df, datetime_cols=None, fillna_cols=None):
    """
    Clean a DataFrame by removing duplicates, converting columns to datetime, and imputing missing values.

    This function performs essential data cleaning operations to prepare data for analysis:
    1. Removes duplicate rows to ensure data integrity
    2. Converts specified columns to datetime format for time-based analysis
    3. Fills missing values using specified strategies (median for numeric, mode for categorical)

    Parameters:
        df (pd.DataFrame): Input DataFrame to clean. Original DataFrame is not modified.
        datetime_cols (list of str, optional): List of column names to convert to datetime format.
            Uses pandas.to_datetime() with error handling. Defaults to None.
        fillna_cols (dict, optional): Dictionary mapping column names to fill strategies.
            Valid strategies: 'median' (for numerical columns) or 'mode' (for categorical columns).
            Defaults to None.

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed, datetime columns converted,
            and missing values imputed according to specified strategies.

    Raises:
        KeyError: If specified columns in datetime_cols or fillna_cols don't exist in DataFrame.
        ValueError: If an invalid fill strategy is provided in fillna_cols.

    Example:
        >>> df_clean = clean_data(
        ...     df,
        ...     datetime_cols=['purchase_time', 'signup_time'],
        ...     fillna_cols={'age': 'median', 'category': 'mode'}
        ... )
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.drop_duplicates()  # Remove duplicate rows to ensure data integrity

    # Convert specified columns to datetime format if provided
    if datetime_cols:
        for col in datetime_cols:
            # Convert specified columns to datetime format with error handling
            # This enables time-based analysis and filtering operations
            df[col] = pd.to_datetime(df[col])

    # Fill missing values using specified strategies if provided
    if fillna_cols:
        for col, strategy in fillna_cols.items():
            if strategy == "median":
                # Fill missing values with median for numerical columns
                # Median is robust to outliers and suitable for skewed distributions
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                # Fill missing values with mode for categorical columns
                # Mode represents the most frequent category in the data
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


def merge_ip_to_country(df, ip_df):
    """
    Merge an e-commerce DataFrame with an IP-to-country mapping DataFrame.

    This function performs IP geolocation by mapping IP addresses to countries
    using IP range lookup. For each IP address in the e-commerce data, it finds
    the corresponding country based on which IP range the address falls into.

    Parameters:
        df (pd.DataFrame): E-commerce DataFrame containing an 'ip_address' column.
            The IP addresses should be convertible to integers (IPv4 numeric format).
        ip_df (pd.DataFrame): IP-to-country mapping DataFrame with required columns:
            - 'lower_bound_ip_address': Lower bound of IP range (int)
            - 'upper_bound_ip_address': Upper bound of IP range (int)
            - 'country': Country name or code (str)

    Returns:
        pd.DataFrame: Original DataFrame with an additional 'country' column.
            Rows with unmappable IP addresses will have 'country' set to 'Unknown'.

    Raises:
        KeyError: If required columns are missing from either DataFrame.
        ValueError: If IP addresses cannot be converted to integers.

    Note:
        This function uses a simple linear search for IP range matching, which may be
        slow for large IP mapping datasets. For better performance with large datasets,
        consider using interval trees or binary search approaches.

    Example:
        >>> ecommerce_df = pd.DataFrame({'ip_address': ['192.168.1.1', '10.0.0.1']})
        >>> ip_mapping_df = pd.DataFrame({
        ...     'lower_bound_ip_address': [3232235776, 167772160],
        ...     'upper_bound_ip_address': [3232236031, 184549375],
        ...     'country': ['Private', 'Private']
        ... })
        >>> result = merge_ip_to_country(ecommerce_df, ip_mapping_df)
    """
    # Ensure IP addresses are in integer format for range comparison
    # This converts string IP addresses to their numeric representation
    df["ip_address"] = df["ip_address"].astype(int)

    def map_ip(ip):
        """
        Inner function to find the country for a given IP address.

        Searches through the IP mapping DataFrame to find which range
        the given IP address falls into.

        Parameters:
            ip (int): IP address as an integer

        Returns:
            str: Country name if found, 'Unknown' if no matching range
        """
        # Find the country for a given IP by checking which range it falls into
        # Uses boolean indexing to find rows where IP falls within the range
        match = ip_df[
            (ip_df["lower_bound_ip_address"] <= ip)
            & (ip_df["upper_bound_ip_address"] >= ip)
        ]
        # Return the country if found, else 'Unknown'
        # iloc[0] gets the first matching country (assumes no overlapping ranges)
        return match["country"].iloc[0] if not match.empty else "Unknown"

    # Apply the mapping function to each IP address in the DataFrame
    # This creates a new 'country' column with the mapped country for each IP
    df["country"] = df["ip_address"].apply(map_ip)

    return df
