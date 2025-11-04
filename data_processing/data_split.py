import pandas as pd

# Load augmented dataset
df = pd.read_csv("data_augmented.csv")

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into validation, test, and training sets
val_df = df.iloc[:20_000]
test_df = df.iloc[20_000:40_000]
train_df = df.iloc[40_000:]

# Save splits
val_df.to_csv("data/val_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)
train_df.to_csv("data/train_data.csv", index=False)
