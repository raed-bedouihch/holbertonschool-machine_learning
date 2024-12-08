#!/usr/bin/env python3
"""a function  that takes a pd.DataFrame and
Sets the Timestamp column as the index of the dataframe."""


def index(df):
    """Returns: the modified pd.DataFrame"""
    return df.set_index("Timestamp")
