#!/usr/bin/env python3
"""a function  that takes a pd.DataFrame and
Sorts it by the High price in descending order."""


def high(df):
    """Returns: the sorted pd.DataFrame"""
    return df.sort_values(by="High", ascending=False)
