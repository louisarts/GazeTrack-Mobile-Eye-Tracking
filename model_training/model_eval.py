import pandas as pd
import numpy as np

train_df = pd.read_csv("data/train_data.csv")

target_train_raw = train_df[['dot_XCam', 'dot_YCam']].values.astype('float32')
target_mean = target_train_raw.mean(axis=0)
target_std = target_train_raw.std(axis=0)

def compute_avg_mae(dataset, model):
    total = 0
    count = 0

    for images, labels_norm in dataset:
        preds_norm = model.predict(images, verbose=0)

        # De-normalize predictions and labels
        preds = preds_norm * target_std + target_mean
        labels = labels_norm.numpy() * target_std + target_mean

        # Compute MAE per sample in original units
        mae = np.mean(np.abs(preds - labels), axis=1)
        total += np.sum(mae)
        count += len(mae)

    avg_mae = total / count
    return avg_mae

