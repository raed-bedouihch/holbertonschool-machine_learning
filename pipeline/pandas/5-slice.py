#!/usr/bin/env python3
"""a function  that takes a pd.DataFrame and
Extracts the columns High, Low, Close, and Volume_BTC
then Selects every 60th row from these columns."""


def slice(df):
    """Returns: the sliced pd.DataFrame"""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
