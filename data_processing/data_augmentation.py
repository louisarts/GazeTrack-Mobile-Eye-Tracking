import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# --- Basic image operations helper functions ---

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def crop_to_box_face(img, x, y, w, h):
    return img.crop((x, y, x + w, y + h))

def crop_to_box_left_eye(img, x, y, w, h):
    return img.crop((x, y, x + w, y + h))

def crop_to_box_right_eye(img, x, y, w, h):
    return img.crop((x, y, x + w, y + h))

def resize_image(img, size=(64, 64)):
    return img.resize(size)

def color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0):
    img = TF.adjust_brightness(img, random.uniform(1 - brightness, 1 + brightness))
    img = TF.adjust_contrast(img, random.uniform(1 - contrast, 1 + contrast))
    img = TF.adjust_saturation(img, random.uniform(1 - saturation, 1 + saturation))
    img = TF.adjust_hue(img, random.uniform(-hue, hue))
    return img

def convert_to_grayscale(img):
    return img.convert("L").convert("RGB")

def add_gaussian_noise(img, mean=0, std=0.01):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std * 255, np_img.shape[:2])
    if np_img.ndim == 3:
        noise = np.stack([noise] * 3, axis=-1)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def to_tensor_image(img): 
    return to_tensor(img)

def show_image(img):
    if isinstance(img, torch.Tensor):
        img = img.clone().detach().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# --- Preprocessing pipelines ---

def preprocess_image_face(image_path, face_x, face_y, face_w, face_h):
    import tensorflow as tf  # delayed import to avoid conflicts
    img = load_image(image_path)
    img = crop_to_box_face(img, face_x, face_y, face_w, face_h)
    img = resize_image(img)
    img = color_jitter(img)
    img = convert_to_grayscale(img)
    img = add_gaussian_noise(img)
    img = to_tensor_image(img).permute(1, 2, 0).numpy().astype('float32')
    return tf.convert_to_tensor(img, dtype=tf.float32)

def preprocess_image_left_eye(image_path, face_x, face_y, face_w, face_h, 
                              left_eye_x, left_eye_y, left_eye_w, left_eye_h):
    import tensorflow as tf
    img = load_image(image_path)
    img = crop_to_box_left_eye(img, left_eye_x + face_x, left_eye_y + face_y, left_eye_w, left_eye_h)
    img = resize_image(img)
    img = color_jitter(img)
    img = convert_to_grayscale(img)
    img = add_gaussian_noise(img)
    img = to_tensor_image(img).permute(1, 2, 0).numpy().astype('float32')
    return tf.convert_to_tensor(img, dtype=tf.float32)

def preprocess_image_right_eye(image_path, face_x, face_y, face_w, face_h, 
                               right_eye_x, right_eye_y, right_eye_w, right_eye_h):
    import tensorflow as tf
    img = load_image(image_path)
    img = crop_to_box_right_eye(img, right_eye_x + face_x, right_eye_y + face_y, right_eye_w, right_eye_h)
    img = resize_image(img)
    img = color_jitter(img)
    img = convert_to_grayscale(img)
    img = add_gaussian_noise(img)
    img = to_tensor_image(img).permute(1, 2, 0).numpy().astype('float32')
    return tf.convert_to_tensor(img, dtype=tf.float32)


# --- Main processing loop ---

df = pd.read_csv("data/data_cleaned.csv")
subfolders = ["face_crops", "l_eye_crops", "r_eye_crops"]

# Ensure crop folders exist
for frame_path in df["frame_path"]:
    base_path = os.path.expanduser(frame_path[:48])
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_path, subfolder), exist_ok=True)

# Process each frame
for i in range(len(df)):
    frame_path = df["frame_path"].iloc[i]

    face_x, face_y, face_w, face_h = df.loc[i, ["face_box_X", "face_box_Y", "face_box_W", "face_box_H"]]
    leye_x, leye_y, leye_w, leye_h = df.loc[i, ["left_eye_X", "left_eye_Y", "left_eye_W", "left_eye_H"]]
    reye_x, reye_y, reye_w, reye_h = df.loc[i, ["right_eye_X", "right_eye_Y", "right_eye_W", "right_eye_H"]]

    # Face crop
    face_img_np = preprocess_image_face(frame_path, face_x, face_y, face_w, face_h).numpy()
    np.savez_compressed(f"{frame_path[:48]}/face_crops/{frame_path[-9:-4]}.npz", image=face_img_np)

    # Left eye crop
    leye_img_np = preprocess_image_left_eye(frame_path, face_x, face_y, face_w, face_h, leye_x, leye_y, leye_w, leye_h).numpy()
    np.savez_compressed(f"{frame_path[:48]}/l_eye_crops/{frame_path[-9:-4]}.npz", image=leye_img_np)

    # Right eye crop
    reye_img_np = preprocess_image_right_eye(frame_path, face_x, face_y, face_w, face_h, reye_x, reye_y, reye_w, reye_h).numpy()
    np.savez_compressed(f"{frame_path[:48]}/r_eye_crops/{frame_path[-9:-4]}.npz", image=reye_img_np)

    os.remove(frame_path)


# --- Update CSV with crop paths ---

face_crop_path = []
l_eye_crop_path = []
r_eye_crop_path = []

for i in df["frame_path"]:
    face_crop_path.append(i.replace("frames", "face_crops").replace(".jpg", ".npz"))
    l_eye_crop_path.append(i.replace("frames", "l_eye_crops").replace(".jpg", ".npz"))
    r_eye_crop_path.append(i.replace("frames", "r_eye_crops").replace(".jpg", ".npz"))

df["face_crop_path"] = [p.replace("/face_crops/", "/f/face_crops/") for p in face_crop_path]
df["l_eye_crop_path"] = [p.replace("/l_eye_crops/", "/f/l_eye_crops/") for p in l_eye_crop_path]
df["r_eye_crop_path"] = [p.replace("/r_eye_crops/", "/f/r_eye_crops/") for p in r_eye_crop_path]

# --- Save updated dataframe ---
df.to_csv("data/data_augmented.csv", index=False)