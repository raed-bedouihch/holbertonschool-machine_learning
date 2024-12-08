#!/usr/bin/env python3
"""a function  that takes a pd.DataFrame and
Sorts the data in reverse chronological order.
then Transposes the sorted dataframe."""


def flip_switch(df):
    """Returns: the transformed pd.DataFrame"""
    return df.sort_values(by="Timestamp", ascending=False).transpose()
