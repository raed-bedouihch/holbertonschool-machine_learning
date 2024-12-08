#!/usr/bin/env python3
"""a function that takes a pd.DataFrame as input and
select the last 10 rows of the High and Close columns
then Convert these selected values into a numpy.ndarray."""


def array(df):
    """Returns: the numpy.ndarray"""
    return (df[["High", "Close"]].iloc[-10:]).to_numpy()
