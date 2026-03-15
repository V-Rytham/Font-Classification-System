# Font Classification System

This project classifies fonts from word images using a TensorFlow CNN. It also includes a small web app so you can upload an image and run inference from the browser.

## What is in this repository?

- `train/` and `valid/`: image datasets organized by font name
- `train.py`: training entrypoint (CLI)
- `utils.py`: preprocessing helpers + CNN builder
- `backend/`: FastAPI inference API
- `frontend/`: static web UI

---

## Local setup (from scratch)

### 1) Clone and enter the repository

```bash
git clone <your-repo-url>
cd Font-Classification-System
```

### 2) Create a virtual environment

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

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## If you do not have `CNN_Font_Classification.h5`

That is expected for a fresh setup. Train it once using the dataset already in this repo.

### 1) Sanity-check dataset folders

```bash
python - <<'PY'
from pathlib import Path
train = Path('train')
valid = Path('valid')
print('train exists:', train.exists())
print('valid exists:', valid.exists())
print('train classes:', len([p for p in train.iterdir() if p.is_dir()]))
print('valid classes:', len([p for p in valid.iterdir() if p.is_dir()]))
PY
```

### 2) Train the model

```bash
python train.py --epochs 6 --train-dir train --valid-dir valid --output-model CNN_Font_Classification.h5
```

This command writes:

- `CNN_Font_Classification.h5`
- `class_names.json`

`class_names.json` is important because it keeps class index mapping consistent during inference.

### 3) (Optional) quick smoke run for low-resource machines

```bash
python train.py --epochs 1 --train-batch-size 64 --valid-batch-size 32
```

---

## Run the web app

Start the API server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open these URLs:

- App UI: `http://localhost:8000`
- Health endpoint: `http://localhost:8000/api/health`

If the health response reports `"status": "ready"`, the model loaded correctly.

---

## API usage

### Health check

```bash
curl http://localhost:8000/api/health
```

### Prediction

```bash
curl -X POST "http://localhost:8000/api/predict?top_k=3" \
  -F "file=@test/advisorily_157.jpg"
```

---

## Common issues

### `status: degraded` in `/api/health`

Usually means the model path is wrong or the model file is missing.

If your model is stored elsewhere:

**macOS/Linux**

```bash
export MODEL_PATH=/absolute/path/to/CNN_Font_Classification.h5
export CLASSES_DIR=/absolute/path/to/train
```

**Windows (PowerShell)**

```powershell
$env:MODEL_PATH="C:\absolute\path\to\CNN_Font_Classification.h5"
$env:CLASSES_DIR="C:\absolute\path\to\train"
```

Then run uvicorn again.

### Training is slow

- reduce epochs (`--epochs 1` for a first pass)
- lower batch sizes
- use GPU TensorFlow if available

---

## Notes

- The backend reads labels from `class_names.json` when present.
- If `class_names.json` is missing, it falls back to reading class folder names from `CLASSES_DIR`.

---

## Credits

- Original project: Shubham Kapoor
- TRDG toolkit: Edouard Belval
