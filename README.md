# Font Classification System

A TensorFlow-based font recognition project with a minimal web application for image upload and font prediction.

## 1) Repository Analysis

This repository contains three major parts:

- **Model training code** (`train.py`, `utils.py`)
- **Single-image prediction script** (`test.py`)
- **Synthetic data generation tooling** (`trdg/`) used to produce font image datasets

### ML framework

The model is implemented with **TensorFlow / Keras** (not PyTorch):

- `utils.py` defines a CNN architecture with convolution, pooling, dropout, batch normalization, and dense output layers.
- Output layer size is configured for **100 font classes**.

### Training and inference flow (original project)

- `train.py`
  - Loads data from `train/` and `valid/` folders via `ImageDataGenerator.flow_from_directory`.
  - Trains the CNN.
  - Saves model as `CNN_Font_Classification.h5`.
- `test.py`
  - Loads one test image.
  - Center-crops/resizes preprocessing to `100x100`.
  - Loads saved model.
  - Predicts class index and maps to class name.

### Dataset layout

The repo already includes class-structured datasets:

- `train/<font_name>/*.jpg`
- `valid/<font_name>/*.jpg`

The class names are inferred from subfolder names.

## 2) Web Application (Added)

A clean, minimal web app is now included.

### New structure

```text
project/
├── backend/
│   ├── inference/
│   │   └── service.py
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── requirements.txt
├── train.py
├── test.py
├── utils.py
└── README.md
```

### Features

- Upload image with text
- Call backend prediction API
- Show top predicted fonts with confidence
- Minimal black/white/gray UI with centered card layout

## 3) Local Setup

## Prerequisites

- Python 3.10+ recommended
- `pip`

## Install dependencies

```bash
pip install -r requirements.txt
```

## Model requirement

The API expects a trained model at:

- `./CNN_Font_Classification.h5` (default)

If your model is elsewhere, set environment variable:

```bash
export MODEL_PATH=/absolute/path/to/CNN_Font_Classification.h5
```

Optional class directory override:

```bash
export CLASSES_DIR=/absolute/path/to/train
```

## Run the web app

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open:

- `http://localhost:8000` for UI
- `http://localhost:8000/api/health` for backend status

## 4) API Usage

### Health check

```bash
curl http://localhost:8000/api/health
```

### Predict font(s)

```bash
curl -X POST "http://localhost:8000/api/predict?top_k=3" \
  -F "file=@test/advisorily_157.jpg"
```

Response format:

```json
{
  "predictions": [
    {"font": "Roboto-Bold", "confidence": 0.913245},
    {"font": "Roboto-Regular", "confidence": 0.046219},
    {"font": "OpenSans-Bold", "confidence": 0.015111}
  ]
}
```

## 5) Training the Model (Original Scripts)

The original scripts in this repository use local hardcoded paths. Update paths in `train.py`/`test.py` or refactor before retraining.

Typical flow:

1. Ensure dataset in `train/` and `valid/`
2. Run training script
3. Save model as `CNN_Font_Classification.h5`
4. Start API and test upload via UI

## 6) Runtime and low-resource notes

- TensorFlow can be heavy on memory/CPU.
- For low-resource deployments:
  - Use CPU builds (`tensorflow` CPU wheel depending on platform).
  - Run with a single API worker.
  - Keep `top_k` small.
  - Consider model quantization/export to TFLite in a future iteration.

## 7) Free hosting options

- **Render** (web service, easy FastAPI deployment)
- **Railway** (simple app deployment)
- **Hugging Face Spaces** (Docker/Gradio/Static options)
- **Fly.io** (containerized deployment)

For best portability, deploy with Docker and set `MODEL_PATH` + `CLASSES_DIR` through environment variables.

## 8) Credits

- Original project: Shubham Kapoor
- Dataset generation tooling: TRDG (Edouard Belval)
