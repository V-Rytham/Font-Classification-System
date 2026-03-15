# Font Classification System

A TensorFlow-based font recognition project with a minimal web application for image upload and font prediction.

## Project at a glance

- **ML framework:** TensorFlow / Keras CNN.
- **Dataset format:** `train/<font_name>/*.jpg` and `valid/<font_name>/*.jpg`.
- **Web app:** FastAPI backend + minimal static frontend.

---

## 1) If you do NOT have `CNN_Font_Classification.h5`

No problem. Train it locally with this exact sequence.

### Step 1 — Open the repo

```bash
cd /path/to/Font-Classification-System
```

### Step 2 — Create a virtual environment

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Confirm dataset exists

You should have:

- `train/` with many font subfolders
- `valid/` with matching font subfolders

Quick check:

```bash
python - <<'PY'
from pathlib import Path
print('train exists:', Path('train').exists())
print('valid exists:', Path('valid').exists())
print('train classes:', len([p for p in Path('train').iterdir() if p.is_dir()]))
print('valid classes:', len([p for p in Path('valid').iterdir() if p.is_dir()]))
PY
```

### Step 5 — Train model (beginner-safe command)

```bash
python train.py --epochs 6 --train-dir train --valid-dir valid --output-model CNN_Font_Classification.h5
```

This creates:

- `CNN_Font_Classification.h5`
- `class_names.json`

> `class_names.json` is used by the API to keep class index mapping correct at inference time.

### Step 6 — Start the API + web app

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Step 7 — Open in browser

- UI: `http://localhost:8000`
- Health: `http://localhost:8000/api/health`

### Step 8 — Predict a font

Use the upload box in UI, or call API:

```bash
curl -X POST "http://localhost:8000/api/predict?top_k=3" \
  -F "file=@test/advisorily_157.jpg"
```

---

## 2) Fast path if you ALREADY have a model file elsewhere

Set `MODEL_PATH` and optionally `CLASSES_DIR`, then start server.

**macOS/Linux**

```bash
export MODEL_PATH=/absolute/path/to/CNN_Font_Classification.h5
export CLASSES_DIR=/absolute/path/to/train
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell)**

```powershell
$env:MODEL_PATH="C:\absolute\path\to\CNN_Font_Classification.h5"
$env:CLASSES_DIR="C:\absolute\path\to\train"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 3) Training script options

```bash
python train.py --help
```

Important flags:

- `--epochs` (default: 6)
- `--train-batch-size` (default: 300)
- `--valid-batch-size` (default: 100)
- `--crop-dataset` (optional, forces all images to `100x100`)
- `--output-model` (default: `CNN_Font_Classification.h5`)
- `--output-labels` (default: `class_names.json`)

---

## 4) Architecture summary

- `utils.py`
  - dataset listing/preprocessing helpers
  - CNN builder (`build_model`)
- `train.py`
  - trains model using folder datasets
  - saves model + label mapping JSON
- `backend/inference/service.py`
  - loads model and labels
  - prepares input image and returns top-k predictions
- `backend/main.py`
  - FastAPI routes (`/api/health`, `/api/predict`)
  - serves frontend
- `frontend/`
  - minimal upload UI and result rendering

---

## 5) Low-resource tips

- Start with fewer epochs: `--epochs 1` for first smoke test.
- Use smaller batch sizes if memory is low:

```bash
python train.py --epochs 1 --train-batch-size 64 --valid-batch-size 32
```

- Keep API single worker (`uvicorn ...` default is fine).

---

## 6) Troubleshooting

### `status: degraded` in `/api/health`

- Model missing or wrong path.
- Fix by training first or setting `MODEL_PATH` correctly.

### Training is too slow

- Reduce epochs and batch size.
- Use GPU TensorFlow if available.

### Wrong labels in output

- Keep `class_names.json` next to the model file.
- If missing, API falls back to directory names from `CLASSES_DIR`.

---

## Credits

- Original project: Shubham Kapoor
- Dataset generation tooling: TRDG (Edouard Belval)
