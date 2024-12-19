import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv("./data/cervical_train.csv")

# Check for any NaN values in the entire dataset
print("Total NaN values in dataset:", df.isna().sum().sum())

# If there are NaN values, show them by column
if df.isna().sum().sum() > 0:
    print("\nNaN values by column:")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0])
