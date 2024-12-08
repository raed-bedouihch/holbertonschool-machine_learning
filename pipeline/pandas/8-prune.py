#!/usr/bin/env python3
"""a function  that takes a pd.DataFrame and
Removes any entries where Close has NaN values."""


def prune(df):
    """Returns: the modified pd.DataFrame"""
    return df.dropna(subset=["Close"])
