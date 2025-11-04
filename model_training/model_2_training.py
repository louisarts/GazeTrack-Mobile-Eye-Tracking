from model_training.model_eval import compute_avg_mae
from model_training.training_helper_functions import (
    arch_2_conv_neural_network,
    save_model,
    create_dataset,
)
import pandas as pd

# Model hyperparameters
HPARAMS = {
    "l2_penalty": 1e-5,
    "conv1_filters": 32,
    "conv2_filters": 64,
    "conv3_filters": 128,
    "dense1_units": 128,
    "dense2_units": 128,
    "learning_rate": 1e-3,
    "training_epochs": 25,
}

# Load training and validation data
train_df = pd.read_csv("../data/train_data.csv")
val_df = pd.read_csv("../data/val_data.csv")

# Build and train model
model, history = arch_2_conv_neural_network(
    HPARAMS["conv1_filters"],
    HPARAMS["conv2_filters"],
    HPARAMS["conv3_filters"],
    HPARAMS["dense1_units"],
    HPARAMS["dense2_units"],
    HPARAMS["learning_rate"],
    HPARAMS["training_epochs"],
)

# Create datasets
train_ds = create_dataset(train_df, batch_size=64)
val_ds = create_dataset(val_df, batch_size=64)

# Evaluate model performance
train_mae = compute_avg_mae(train_ds, model)
val_mae = compute_avg_mae(val_ds, model)

print(f"Train MAE: {train_mae}")
print(f"Val MAE: {val_mae}")

# Save model and training history
save_model(model, "../models/model_2")
history_2_df = pd.DataFrame(history.history)
history_2_df.to_csv("../training_histories/model_2_history.csv", index=False)
