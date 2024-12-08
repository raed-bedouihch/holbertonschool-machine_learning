#!/usr/bin/env python3
"""a script to visualize the pd.DataFrame"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=["Weighted_Price"])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df['Date'], unit='s')
df.set_index("Date", inplace=True)
df["Close"] = df["Close"].fillna(method="ffill")
for col in ["High", "Low", "Open"]:
    df[col] = df[col].fillna(df["Close"])
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

df = df[df.index.year >= 2017]

df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})
plt.figure(figsize=(10, 6))
plt.plot(df_daily, label=df_daily.columns)
plt.xlabel('Date')
x_ticks = df_daily.index[df_daily.index.month.isin([1, 4, 7, 10])]
x_labels = x_ticks.strftime('%b').tolist()
for i in range(len(x_labels)):
    if x_ticks[i].month == 1:
        x_labels[i] = x_ticks[i].strftime('%b %Y')
    else:
        x_labels[i] = x_ticks[i].strftime('%b %Y')
plt.xticks(x_ticks, x_labels, rotation=45)
# print(x_ticks)
# print(x_labels)
plt.legend()
plt.tight_layout()
print(df_daily)
plt.show()
