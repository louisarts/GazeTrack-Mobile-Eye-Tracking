import io
import time
import base64
from typing import Optional
from flask import Flask, request, jsonify
import numpy as np
import cv2
import model  # must have predict_coords(np_img: np.ndarray) -> [x, y]

app = Flask(__name__, static_url_path="", static_folder="static")

def _decode_image_to_bgr(data: bytes) -> Optional[np.ndarray]:
    # decode image bytes (JPEG/PNG) into a BGR NumPy array for OpenCV
    try:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except Exception:
        return None

@app.route("/")
def index():
    # serve the index page from the static folder
    return app.send_static_file("index.html")

@app.post("/predict")
def predict():
    # receive an image from form-data, raw body, or base64 JSON
    image_bytes = None

    if "frame" in request.files:
        image_bytes = request.files["frame"].read()
    elif request.data:
        image_bytes = request.data
    else:
        try:
            payload = request.get_json(force=True, silent=True) or {}
            if "image" in payload and isinstance(payload["image"], str):
                b64 = payload["image"]
                if b64.startswith("data:") and "," in b64:
                    b64 = b64.split(",", 1)[1]
                image_bytes = base64.b64decode(b64)
        except Exception:
            image_bytes = None

    # return error if no image received
    if not image_bytes:
        return jsonify({"ok": False, "error": "No image received"}), 400

    # decode the image into a NumPy array
    np_img_bgr = _decode_image_to_bgr(image_bytes)
    if np_img_bgr is None:
        return jsonify({"ok": False, "error": "Could not decode image"}), 400

    # run the model prediction
    coords = model.predict_coords(np_img_bgr)
    if coords is None:
        return jsonify({"ok": True, "coords": None})

    # ensure the model output is valid
    try:
        x, y = float(coords[0]), float(coords[1])
    except Exception:
        return jsonify({"ok": False, "error": "Model returned invalid coords"}), 500

    # include image shape and timestamp in response
    h, w = np_img_bgr.shape[:2]
    return jsonify({"ok": True, "coords": [x, y], "ts": time.time(), "shape": [int(h), int(w)]})

if __name__ == "__main__":
    # run HTTPS server with an auto-generated self-signed certificate
    # access it via https://<your-laptop-ip>:8000
    app.run(host="0.0.0.0", port=8000, debug=False, ssl_context="adhoc")
    # for local debugging without HTTPS, uncomment below
    # app.run(host="0.0.0.0", port=8000, debug=True)
