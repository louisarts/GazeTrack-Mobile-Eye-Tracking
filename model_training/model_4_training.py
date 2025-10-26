from model_training.model_eval import compute_avg_mae
from model_training.training_helper_functions import arch_3_conv_neural_network, save_model, create_dataset
import pandas as pd


HPARAMS = {"l2_penalty":1e-6,
          "conv1_filters":32,
          "conv2_filters":64,
          "conv3_filters":128,
          "dense1_units":128,
          "dense2_units":128,
          "dropout_rate":0.3,
          "learning_rate":1e-3,
          "training_epochs":25}

train_df = pd.read_csv("../data/train_data.csv")
val_df = pd.read_csv("../data/val_data.csv")

model, history = arch_3_conv_neural_network(HPARAMS["conv1_filters"],
                                     HPARAMS["conv2_filters"],
                                     HPARAMS["conv3_filters"],
                                     HPARAMS["dense1_units"],
                                     HPARAMS["dense2_units"],
                                     HPARAMS["learning_rate"],
                                     HPARAMS["training_epochs"])

train_ds = create_dataset(train_df, batch_size=64)
val_ds = create_dataset(val_df, batch_size=64)

train_mae = compute_avg_mae(train_ds, model)
val_mae = compute_avg_mae(val_ds, model)

print(f"Train mae: {train_mae}")
print(f"Val mae: {val_mae}")

save_model(model, "../models/model_4")

history_1_df = pd.DataFrame(history.history)
history_1_df.to_csv("../training_histories/model_4_history.csv", index=False)