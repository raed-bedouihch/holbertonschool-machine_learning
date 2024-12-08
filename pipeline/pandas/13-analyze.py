#!/usr/bin/env python3
"""a function  that takes a pd.DataFrame and
Computes descriptive statistics for all columns
except the Timestamp column"""


def analyze(df):
    """Returns a new pd.DataFrame containing these statistics."""
    new_df = df.drop(columns=["Timestamp"])
    return new_df.describe()
