import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization


train_df = pd.read_csv("data/train_data.csv")
val_df = pd.read_csv("data/val_data.csv")

# --- defining training data mean and standard deviation for target normalisation ---
target_train_raw = train_df[['dot_XCam', 'dot_YCam']].values.astype('float32')
target_mean = target_train_raw.mean(axis=0)
target_std = target_train_raw.std(axis=0)

# Convert to tf constants once so they can be used in metrics
target_mean_tf = tf.constant(target_mean, dtype=tf.float32)
target_std_tf = tf.constant(target_std, dtype=tf.float32)

# --- Helper functions
def load_npz_tf(face_path, l_eye_path, r_eye_path, label):
    def _load(path_tensor):
        path_str = path_tensor.numpy().decode("utf-8")
        arr = np.load(path_str)['image']
        return arr.astype(np.float32)

    face = tf.py_function(_load, [face_path], tf.float32)
    l_eye = tf.py_function(_load, [l_eye_path], tf.float32)
    r_eye = tf.py_function(_load, [r_eye_path], tf.float32)

    face.set_shape([64, 64, 3])
    l_eye.set_shape([64, 64, 3])
    r_eye.set_shape([64, 64, 3])
    
    # normalize labels
    label = (label - tf.constant(target_mean, tf.float32)) / tf.constant(target_std, tf.float32)

    return (face, l_eye, r_eye), label


def create_dataset(df, batch_size=64):
    face_paths = df['face_crop_path'].values
    l_eye_paths = df['l_eye_crop_path'].values
    r_eye_paths = df['r_eye_crop_path'].values
    labels = df[['dot_XCam', 'dot_YCam']].values.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((face_paths, l_eye_paths, r_eye_paths, labels))
    ds = ds.map(load_npz_tf, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- loading train and validation from csv ---
train_ds = create_dataset(train_df, batch_size=64)
val_ds = create_dataset(val_df, batch_size=64)


# --- custom metric for denormalized MAE ---
def denorm_mae(y_true, y_pred):
    y_true_denorm = y_true * target_std_tf + target_mean_tf
    y_pred_denorm = y_pred * target_std_tf + target_mean_tf
    return tf.reduce_mean(tf.abs(y_true_denorm - y_pred_denorm))

# --- model ---

def arch_1_conv_neural_network( conv1_filters, 
                        conv2_filters, 
                        conv3_filters, 
                        dense1_units,
                        dense2_units,
                        learning_rate,
                        training_epochs):
    
    
    
    face_input = Input(shape=(64, 64, 3), name="face_input")
    x_face = layers.Conv2D(conv1_filters, 3, activation='relu')(face_input)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv2_filters, 3, activation='relu')(x_face)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv3_filters, 3, activation='relu')(x_face)
    x_face = layers.Flatten()(x_face)
    x_face = layers.Dense(dense1_units, activation='relu')(x_face)
    
    l_eye_input = Input(shape=(64, 64, 3), name="left_eye_input")
    x_left = layers.Conv2D(conv1_filters, 3, activation='relu')(l_eye_input)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv2_filters, 3, activation='relu')(x_left)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv3_filters, 3, activation='relu')(x_left)
    x_left = layers.Flatten()(x_left)
    x_left = layers.Dense(dense1_units, activation='relu')(x_left)

    r_eye_input = Input(shape=(64, 64, 3), name="right_eye_input")
    x_right = layers.Conv2D(conv1_filters, 3, activation='relu')(r_eye_input)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv2_filters, 3, activation='relu')(x_right)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv3_filters, 3, activation='relu')(x_right)
    x_right = layers.Flatten()(x_right)
    x_right = layers.Dense(dense1_units, activation='relu')(x_right)
    
    combined = layers.Concatenate()([x_face, x_left, x_right])
    x = layers.Dense(dense2_units, activation='relu')(combined)
    output = layers.Dense(2)(x)
    
    model = Model(inputs=[face_input, l_eye_input, r_eye_input], outputs=output)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',                   # still in normalized space
    metrics=[denorm_mae]          # MAE in de-normalized space
)
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_epochs
)
    
    return model, history


def arch_2_conv_neural_network(l2_penalty, 
                        conv1_filters, 
                        conv2_filters, 
                        conv3_filters, 
                        dense1_units,
                        dense2_units,
                        learning_rate,
                        training_epochs):
    
    reg = regularizers.l2(l2_penalty)
    
    face_input = Input(shape=(64, 64, 3), name="face_input")
    x_face = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(face_input)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.Flatten()(x_face)
    x_face = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_face)
    
    l_eye_input = Input(shape=(64, 64, 3), name="left_eye_input")
    x_left = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(l_eye_input)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.Flatten()(x_left)
    x_left = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_left)

    r_eye_input = Input(shape=(64, 64, 3), name="right_eye_input")
    x_right = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(r_eye_input)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.Flatten()(x_right)
    x_right = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_right)
    
    combined = layers.Concatenate()([x_face, x_left, x_right])
    x = layers.Dense(dense2_units, activation='relu', kernel_regularizer=reg)(combined)
    output = layers.Dense(2, kernel_regularizer=reg)(x)
    
    model = Model(inputs=[face_input, l_eye_input, r_eye_input], outputs=output)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',                   # still in normalized space
    metrics=[denorm_mae]          # MAE in de-normalized space
)

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_epochs
)
    
    return model, history

def arch_3_conv_neural_network(l2_penalty, 
                        conv1_filters, 
                        conv2_filters, 
                        conv3_filters, 
                        dense1_units,
                        dense2_units,
                        dropout_rate,
                        learning_rate,
                        training_epochs):
    
    reg = regularizers.l2(l2_penalty)
    
    face_input = Input(shape=(64, 64, 3), name="face_input")
    x_face = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(face_input)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.Flatten()(x_face)
    x_face = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.Dropout(dropout_rate)(x_face)
    
    l_eye_input = Input(shape=(64, 64, 3), name="left_eye_input")
    x_left = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(l_eye_input)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.Flatten()(x_left)
    x_left = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.Dropout(dropout_rate)(x_left)
    
    r_eye_input = Input(shape=(64, 64, 3), name="right_eye_input")
    x_right = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(r_eye_input)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.Flatten()(x_right)
    x_right = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.Dropout(dropout_rate)(x_right)

    combined = layers.Concatenate()([x_face, x_left, x_right])
    x = layers.Dense(dense2_units, activation='relu', kernel_regularizer=reg)(combined)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(2, kernel_regularizer=reg)(x)
    
    model = Model(inputs=[face_input, l_eye_input, r_eye_input], outputs=output)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',                   # still in normalized space
    metrics=[denorm_mae]          # MAE in de-normalized space
)

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_epochs
)
    
    return model, history

def arch_4_conv_neural_network(l2_penalty, 
                        conv1_filters, 
                        conv2_filters, 
                        conv3_filters, 
                        dense1_units,
                        dense2_units,
                        dropout_rate,
                        learning_rate,
                        earlystop_patience,
                        training_epochs):
    
    reg = regularizers.l2(l2_penalty)
    
    face_input = Input(shape=(64, 64, 3), name="face_input")
    x_face = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(face_input)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.Flatten()(x_face)
    x_face = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.Dropout(dropout_rate)(x_face)
    
    l_eye_input = Input(shape=(64, 64, 3), name="left_eye_input")
    x_left = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(l_eye_input)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.Flatten()(x_left)
    x_left = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.Dropout(dropout_rate)(x_left)
    
    r_eye_input = Input(shape=(64, 64, 3), name="right_eye_input")
    x_right = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(r_eye_input)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.Flatten()(x_right)
    x_right = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.Dropout(dropout_rate)(x_right)

    combined = layers.Concatenate()([x_face, x_left, x_right])
    x = layers.Dense(dense2_units, activation='relu', kernel_regularizer=reg)(combined)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(2, kernel_regularizer=reg)(x)
    
    model = Model(inputs=[face_input, l_eye_input, r_eye_input], outputs=output)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',                   # still in normalized space
    metrics=[denorm_mae]          # MAE in de-normalized space
)

    early_stop = EarlyStopping(
    monitor='val_loss', patience=earlystop_patience, restore_best_weights=True, verbose=1
)

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_epochs,
    callbacks=[early_stop]
)
    
    return model, history

def arch_5_conv_neural_network(l2_penalty, 
                        conv1_filters, 
                        conv2_filters, 
                        conv3_filters, 
                        dense1_units,
                        dense2_units,
                        dropout_rate,
                        learning_rate,
                        earlystop_patience,
                        training_epochs):
    
    reg = regularizers.l2(l2_penalty)
    
    face_input = Input(shape=(64, 64, 3), name="face_input")
    x_face = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(face_input)
    x_face = BatchNormalization()(x_face)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = BatchNormalization()(x_face)
    x_face = layers.MaxPooling2D()(x_face)
    x_face = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = BatchNormalization()(x_face)
    x_face = layers.Flatten()(x_face)
    x_face = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_face)
    x_face = layers.Dropout(dropout_rate)(x_face)
    
    l_eye_input = Input(shape=(64, 64, 3), name="left_eye_input")
    x_left = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(l_eye_input)
    x_left = BatchNormalization()(x_left)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = BatchNormalization()(x_left)
    x_left = layers.MaxPooling2D()(x_left)
    x_left = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = BatchNormalization()(x_left)
    x_left = layers.Flatten()(x_left)
    x_left = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_left)
    x_left = layers.Dropout(dropout_rate)(x_left)
    
    r_eye_input = Input(shape=(64, 64, 3), name="right_eye_input")
    x_right = layers.Conv2D(conv1_filters, 3, activation='relu', kernel_regularizer=reg)(r_eye_input)
    x_right = BatchNormalization()(x_right)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv2_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = BatchNormalization()(x_right)
    x_right = layers.MaxPooling2D()(x_right)
    x_right = layers.Conv2D(conv3_filters, 3, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = BatchNormalization()(x_right)
    x_right = layers.Flatten()(x_right)
    x_right = layers.Dense(dense1_units, activation='relu', kernel_regularizer=reg)(x_right)
    x_right = layers.Dropout(dropout_rate)(x_right)

    combined = layers.Concatenate()([x_face, x_left, x_right])
    x = layers.Dense(dense2_units, activation='relu', kernel_regularizer=reg)(combined)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(2, kernel_regularizer=reg)(x)
    
    model = Model(inputs=[face_input, l_eye_input, r_eye_input], outputs=output)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',                   # still in normalized space
    metrics=[denorm_mae]          # MAE in de-normalized space
)

    early_stop = EarlyStopping(
    monitor='val_loss', patience=earlystop_patience, restore_best_weights=True, verbose=1
)

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_epochs,
    callbacks=[early_stop]
)
    
    return model, history


def save_model(model, model_name):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = f"models/{model_name}.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Model saved to: {tflite_path}")
    return tflite_path




