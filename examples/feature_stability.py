"""
Analyze feature stability over time
"""

import pandas as pd
import matplotlib.pyplot as plt

channels = ['C3', 'C4']

window_seconds = 2.5
data = pd.read_csv(f"recorded_features_right.csv", header=None)
# data = data.iloc[:, 0:]
data.plot()
# plt.title(f"Feature stability for window size {window_seconds} seconds")
plt.xlabel("Sample index (Motor Imagery Right)")
plt.ylabel("PSD (log10)")
plt.legend([f"{i}" for i in channels])
plt.savefig(f"recorded_features_stability_right.pdf")
print("length of data:", len(data))
var = data.var(axis=0)
# print(data.head(5))
print(var)
