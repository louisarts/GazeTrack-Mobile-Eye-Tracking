import pandas as pd

df = pd.read_csv("data_augmented.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

val_df = df.iloc[:20_000]
test_df = df.iloc[20_000:40_000]
train_df = df.iloc[40_000:]

val_df.to_csv("data/val_data.csv")
test_df.to_csv("data/test_data.csv")
train_df.to_csv("data/train_data.csv")

