#!/usr/bin/env python3
"""a function that takes a pd.DataFrame as input and Convert
the timestamp values to datatime values and Display only
the Datetime and Close column"""
import pandas as pd


def rename(df):
    """Returns: the modified pd.DataFrame"""
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
