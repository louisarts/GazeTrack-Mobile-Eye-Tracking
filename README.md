# GazeTrack Mobile Eye Tracking

A full pipeline for training gaze-estimation convolutional neural networks on the GazeCapture dataset and serving the best-performing TensorFlow Lite model behind a lightweight mobile-friendly Flask application.

## Table of contents
1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Data processing pipeline](#data-processing-pipeline)
4. [Model training scripts](#model-training-scripts)
5. [Producing your own tuned model](#producing-your-own-tuned-model)
6. [Running the mobile web application](#running-the-mobile-web-application)
7. [Artifacts](#artifacts)
8. [License](#license)

## Quick start
```bash
# create & activate a virtual environment (Python 3.9+ recommended)
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

* Set `RAW_DATA_PATH` in [`src/config.py`](src/config.py) to the directory where you will download the raw GazeCapture archives.
* Follow the [Producing your own tuned model](#producing-your-own-tuned-model) section to process data and train.
* Use the [Running the mobile web application](#running-the-mobile-web-application) section to serve predictions from a trained TFLite model.

## Repository layout

| Path | Description |
| ---- | ----------- |
| [`LICENSE`](LICENSE) | MIT License for the project. |
| [`README.md`](README.md) | You are here. Project overview, setup, and usage instructions. |
| [`src/config.py`](src/config.py) | Central hyperparameters and the `RAW_DATA_PATH` placeholder that must point at your local GazeCapture root before running any preprocessing scripts. |
| [`data_processing/data_loading.py`](data_processing/data_loading.py) | Extracts downloaded GazeCapture `.tar.gz` archives, parses Apple JSON annotations for faces, eyes, dots, and screen metadata, and compiles `data/raw_data.csv`. |
| [`data_processing/data_cleaning.py`](data_processing/data_cleaning.py) | Filters out non-upright screen orientations and rows with invalid numeric values while deleting problematic frame images, producing `data/data_cleaned.csv`. |
| [`data_processing/data_augmentation.py`](data_processing/data_augmentation.py) | Crops face and eye regions, applies augmentation (jitter, grayscale, Gaussian noise), serialises crops as `.npz`, deletes original frames, and records crop paths in `data/data_augmented.csv`. |
| [`data_processing/data_split.py`](data_processing/data_split.py) | Shuffles the augmented dataset and writes deterministic `data/train_data.csv`, `data/val_data.csv`, and `data/test_data.csv` splits. |
| [`model_training/training_helper_functions.py`](model_training/training_helper_functions.py) | Shared TensorFlow utilities: dataset builders, label normalisation, denormalised MAE metric, five CNN architectures (baseline, L2-regularised, dropout, early-stopping, batch-normalised), and a helper to export trained models to TFLite. |
| [`model_training/model_1_training.py`](model_training/model_1_training.py) | Trains the baseline architecture (no regularisation) and saves weights and training history. |
| [`model_training/model_2_training.py`](model_training/model_2_training.py) | Trains the L2-regularized variant defined in `training_helper_functions.py`. |
| [`model_training/model_3_training.py`](model_training/model_3_training.py) | Trains the dropout-regularized architecture. |
| [`model_training/model_4_training.py`](model_training/model_4_training.py) | Trains the early-stopping variant with dropout and L2 regularization. |
| [`model_training/model_5_training.py`](model_training/model_5_training.py) | Trains the batch-normalized + regularized architecture (default production choice). |
| [`model_training/model_eval.py`](model_training/model_eval.py) | Provides `compute_avg_mae` to report denormalised MAE for any trained Keras model on a dataset iterator. |
| [`mobile_application/app.py`](mobile_application/app.py) | Flask app exposing `/` for the static single-page calibration UI and `/predict` for inference against a module named `model` that provides `predict_coords(np_img)`. |
| [`mobile_application/model_in_prod.py`](mobile_application/model_in_prod.py) | Production inference logic: OpenCV face/eye detection, preprocessing to match training augments, TensorFlow Lite interpreter execution, and denormalisation back to pixel coordinates. Rename or symlink to `mobile_application/model.py` (or adjust `PYTHONPATH`) so `app.py` can import it as `model`. |
| [`mobile_application/static/index.html`](mobile_application/static/index.html) | Mobile-first calibration UI: captures camera frames, sends them to `/predict`, guides the user through a four-corner calibration routine, smooths predictions, and visualises the live gaze dot. |
| [`models/`](models) | Pre-trained TensorFlow Lite models (`model_1.tflite` … `model_5.tflite`). Copy the desired file next to `mobile_application/model.py` for serving. |
| [`training_histories/`](training_histories) | CSV logs (`model_1_history.csv` … `model_5_history.csv`) containing Keras training metrics for analysis and comparisons. |

## Data processing pipeline
1. **Download the dataset** – Visit the [GazeCapture download page](https://gazecapture.csail.mit.edu/download.php), request access, and download the participant archives into `RAW_DATA_PATH/gazecapture/`.
2. **Extract & index** – Run `python data_processing/data_loading.py` to extract `.tar.gz` files into `RAW_DATA_PATH/gazecapture_data/` and generate `data/raw_data.csv` summarising all frames and metadata.
3. **Clean invalid samples** – Run `python data_processing/data_cleaning.py` to drop samples with non-upright orientation or invalid numeric values. Outputs `data/data_cleaned.csv`.
4. **Augment & crop** – Run `python data_processing/data_augmentation.py` to crop face/eye regions, apply augmentations, save `.npz` tensors, and produce `data/data_augmented.csv` with new crop paths.
5. **Create splits** – Run `python data_processing/data_split.py` to shuffle and write train/validation/test CSV splits inside `data/`.

> **Note:** The preprocessing scripts expect large amounts of disk space and will delete original frame JPGs after saving crops. Keep backups of the raw archives if you need to re-run the pipeline.

## Model training scripts
* All architectures share the same multi-input design (face + left eye + right eye) and the denormalized MAE metric from `training_helper_functions.py`.
* Each `model_X_training.py` script:
  1. Reads the split CSVs from `data/`.
  2. Instantiates the corresponding architecture helper.
  3. Trains for the configured epochs (defaults come from `src/config.py`).
  4. Computes train/validation MAE via `model_training/model_eval.py`.
  5. Saves a TensorFlow Lite model under `models/model_X.tflite` and writes a CSV history to `training_histories/model_X_history.csv`.

Adjust the hyperparameters either directly in each script or by editing the shared values inside [`src/config.py`](src/config.py) before training.

## Producing your own tuned model
1. **Download & configure paths**
   * Obtain the raw dataset from [GazeCapture](https://gazecapture.csail.mit.edu/download.php).
   * Extract the archives to a folder such as `/data/gazecapture/`.
   * Update [`src/config.py`](src/config.py) so `RAW_DATA_PATH` points to that folder (for example, `RAW_DATA_PATH = "/data"`).
2. **Prepare the environment**
   * Install dependencies in a virtual environment as shown in [Quick start](#quick-start).
3. **Run preprocessing**
   * Execute the scripts listed in the [Data processing pipeline](#data-processing-pipeline) section in order.
   * Verify that `data/train_data.csv`, `data/val_data.csv`, and `data/test_data.csv` exist.
4. **Train models**
   * Choose one of the scripts in `model_training/` (e.g. `python model_training/model_5_training.py`).
   * Monitor console output for MAE metrics. Check `training_histories/model_5_history.csv` for detailed logs.
5. **Evaluate and iterate**
   * Use `model_training/model_eval.py` with the generated `tf.data` pipelines to compare MAE across models or validation subsets.
   * Tweak hyperparameters (filters, dense units, dropout, learning rate, epochs) in `src/config.py` or the individual training script and re-run training to explore trade-offs.
6. **Export & select**
   * Each training run saves a `.tflite` model under `models/`. Rename or copy the best file to `mobile_application/model.tflite` (or update `model_in_prod.py` to reference the chosen filename).

## Running the mobile web application
1. **Set up the model module**
   * Ensure the TFLite model you want to serve is accessible (e.g. `models/model_5.tflite`).
   * Copy or symlink [`mobile_application/model_in_prod.py`](mobile_application/model_in_prod.py) to `mobile_application/model.py`, or add the repository root to `PYTHONPATH` so `app.py` can import it as `model`.
   * Optionally edit `predict_coords` in `model_in_prod.py` to point at a different `.tflite` file or adjust normalisation constants.
2. **Install runtime dependencies**
   ```bash
   pip install flask opencv-python pillow tensorflow torch torchvision numpy
   ```
3. **Start the server**
   ```bash
   cd mobile_application
   python app.py
   ```
   The app runs on `https://0.0.0.0:8000` with a self-signed certificate. For plain HTTP during development, uncomment the alternative `app.run` call at the bottom of `app.py`.
4. **Use the UI**
   * Visit the server from your mobile device (accept the warnings if needed).
   * Tap to start the camera stream, follow the four-corner calibration prompts, and observe the gaze dot tracking in real time!!

## Artifacts
* **Models** – TensorFlow Lite exports in [`models/`](models) ready for deployment.
* **Training histories** – CSV logs in [`training_histories/`](training_histories) for plotting or comparing experiments.

## License
This project is released under the [MIT License](LICENSE).

---

## Dataset Acknowledgement

This project uses data from the [**GazeCapture dataset**](https://gazecapture.csail.mit.edu/) and the associated **iTracker models** developed at the Massachusetts Institute of Technology (MIT) Computer Science and Artificial Intelligence Laboratory (CSAIL).

**Copyright © 2017** — Kyle Krafka, Aditya Khosla, Petr Kellnhofer, Harini Kannan, Suchendra Bhandarkar, Wojciech Matusik, and Antonio Torralba.  
All rights reserved.

Use of the GazeCapture database and iTracker models is governed by the **“License Agreement for Use of GazeCapture Database and iTracker Models”** provided by MIT CSAIL.  
They are made available **for research and educational use only** and **may not be redistributed or used in commercial applications**.

**Required citation:**  
> Krafka, K., Khosla, A., Kellnhofer, P., Kannan, H., Bhandarkar, S., Matusik, W., & Torralba, A. (2016). *Eye Tracking for Everyone.* CVPR 2016.

This repository does **not** include or redistribute any portion of the GazeCapture data or iTracker models.  
All code is independently implemented and released under the MIT License for educational and non-commercial research purposes.

---

