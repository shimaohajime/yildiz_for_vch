import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('../data/block1_conv1_pca.csv')
columns_to_keep = ["artist_revised", "date_revised", "0", "1"]
df = df[columns_to_keep]

# Drop rows with missing values
initial_size = df.shape[0]
df_cleaned = df.dropna()
dropped_rows = initial_size - df_cleaned.shape[0]
print(f"Dropped {dropped_rows} rows with missing values.")

# Convert 'year' to integer
# Remove rows if date_revised is not a number
df_cleaned = df_cleaned[pd.to_numeric(df_cleaned["date_revised"], errors='coerce').notnull()]
df_cleaned["date_revised"] = df_cleaned["date_revised"].astype(int)

# Find the earliest and latest years in the dataset
min_year = df_cleaned["date_revised"].min()
max_year = df_cleaned["date_revised"].max()
print(f"Earliest year: {min_year}, Latest year: {max_year}")

# Find the most common artist
artist_counts = df_cleaned["artist_revised"].value_counts()
top_100_artists = artist_counts.head(100).index
print(f"Top 100 artists: {top_100_artists.tolist()}")
df_top_100 = df_cleaned[df_cleaned["artist_revised"].isin(top_100_artists)]

# Normalize data for stndardization
df_top_100["0"] = (df_top_100["0"] - df_top_100["0"].mean()) / df_top_100["0"].std()
df_top_100["1"] = (df_top_100["1"] - df_top_100["1"].mean()) / df_top_100["1"].std()

# Save the data in NPSDE format
df_npsde_top100 = df_top_100[["artist_revised","date_revised", "0", "1"]].copy()
df_npsde_top100.rename(columns={"artist_revised":"Label","date_revised": "Time", "0": "x1", "1": "x2"}, inplace=True)
## Sort by artist then by time
df_npsde_top100.sort_values(by=["Label", "Time"], inplace=True)

df_npsde_top100.to_csv("../data/Artist_npsde_top100.csv", index=False)