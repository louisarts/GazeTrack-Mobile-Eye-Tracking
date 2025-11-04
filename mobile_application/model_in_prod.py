import os
import random
import warnings
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor

#Suppress TensorFlow logs and deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")

#Load Haar cascade detectors for faces and eyes
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def load_image(path):
    #Load an image from disk as RGB
    return Image.open(path).convert("RGB")

def resize_image(img, size=(64, 64)):
    #Resize image to given size
    return img.resize(size)

def color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0):
    #Apply random brightness, contrast, saturation, and hue changes
    img = TF.adjust_brightness(img, random.uniform(1 - brightness, 1 + brightness))
    img = TF.adjust_contrast(img, random.uniform(1 - contrast, 1 + contrast))
    img = TF.adjust_saturation(img, random.uniform(1 - saturation, 1 + saturation))
    img = TF.adjust_hue(img, random.uniform(-hue, hue))
    return img

def convert_to_grayscale(img):
    #Convert image to grayscale and back to RGB
    return img.convert("L").convert("RGB")

def add_gaussian_noise(img, mean=0, std=0.01):
    #Add Gaussian noise to the image
    np_img = np.array(img, dtype=np.float32)
    noise = np.random.normal(mean, std * 255, np_img.shape[:2])
    if np_img.ndim == 3:
        noise = np.stack([noise] * 3, axis=-1)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def preprocess_image(img):
    #Apply basic preprocessing: resize, jitter, grayscale, noise, and convert to tensor
    if img is None:
        return None
    img = resize_image(color_jitter(convert_to_grayscale(add_gaussian_noise(img))))
    img = to_tensor(img).permute(1, 2, 0).numpy().astype("float32")
    return tf.convert_to_tensor(img, dtype=tf.float32)


def _detect_face(np_img):
    #Detect the largest face in the image
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return max(faces, key=lambda r: r[2] * r[3]) if len(faces) else None

def _detect_eyes(roi_gray):
    #Detect eyes inside a face region
    eyes = _eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
    return sorted(eyes, key=lambda e: e[0]) if len(eyes) >= 2 else None

def load_crops(np_img):
    #Extract face, left eye, and right eye crops as PIL images
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    face = _detect_face(np_img)
    if face is None:
        return None, None, None

    x, y, w, h = face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = np_img[y:y+h, x:x+w]
    eyes = _detect_eyes(roi_gray)
    if eyes is None:
        return None, None, None

    face_crop = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
    left_crop = Image.fromarray(cv2.cvtColor(
        roi_color[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]], cv2.COLOR_BGR2RGB))
    right_crop = Image.fromarray(cv2.cvtColor(
        roi_color[eyes[1][1]:eyes[1][1]+eyes[1][3], eyes[1][0]:eyes[1][0]+eyes[1][2]], cv2.COLOR_BGR2RGB))

    return face_crop, left_crop, right_crop


def predict_coords(np_img, model_path="model_5.tflite"):
    #Run the TFLite model and return predicted (x, y) pixel coordinates

    #Normalization constants
    x_mean, x_std = 0.52776694, 2.0625846
    y_mean, y_std = -6.493749, 3.7171712

    #Load and preprocess face and eyes
    face, left, right = load_crops(np_img)
    crops = [preprocess_image(c) for c in (face, left, right)]
    if any(c is None for c in crops):
        return None

    #Add batch dimension to inputs
    inputs = [tf.expand_dims(c, 0).numpy() for c in crops]

    #Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Set inputs and run inference
    for i, inp in enumerate(inputs):
        interpreter.set_tensor(input_details[i]["index"], inp)
    interpreter.invoke()

    #Get model outputs
    coords = interpreter.get_tensor(output_details[0]["index"])[0]

    #Convert from normalized values to pixel coordinates
    x_pixels = (coords[0] * x_std + x_mean) * 181
    y_pixels = (coords[1] * y_std + y_mean) * 181

    return [-x_pixels, y_pixels]
