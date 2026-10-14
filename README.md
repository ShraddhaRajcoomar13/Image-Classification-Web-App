# CIFAR-10 Image Classification Web App

A Flask web application that classifies images into one of 10 CIFAR-10 categories using a convolutional neural network built with PyTorch.

---

## Features

- Custom CNN trained on the CIFAR-10 dataset
- Interactive drag-and-drop web UI
- Returns top-3 predictions with confidence scores
- Runs on CPU (no GPU required)
- Simple REST API (`/predict`, `/health`)

---

## Supported Classes

| Label | Emoji |
|-------|-------|
| airplane | ✈️ |
| automobile | 🚗 |
| bird | 🐦 |
| cat | 🐱 |
| deer | 🦌 |
| dog | 🐕 |
| frog | 🐸 |
| horse | 🐴 |
| ship | 🚢 |
| truck | 🚚 |

---

## Project Structure

```
Image-Classification-Web-App/
├── data/                       # CIFAR-10 dataset (downloaded automatically)
├── templates/
│   └── index.html              # Web UI
├── cifar10cnn.py               # CNN model definition
├── train_model.py              # Training loop
├── train_once.py               # Script to train and save the model
├── load_model.py               # Loads checkpoint from disk
├── predict_image.py            # Inference logic
├── image_preprocessing.py      # Image transforms (resize, normalize)
├── flask_app.py                # Flask application
├── config.py                   # Class names and shared config
├── main.py                     # Optional entry point for training
├── test_train.py               # Quick pipeline smoke test
├── requirements.txt
├── runtime.txt
└── cifar10_cnn.pth             # Trained model checkpoint (generated)
```

---

## Requirements

- Python 3.10+ (3.13 tested locally)
- pip

Dependencies (see `requirements.txt`):
```
Flask
torch
torchvision
Pillow
gunicorn
numpy
```

---

## Setup

### 1. Clone / navigate to the project

```powershell
cd Image-Classification-Web-App
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Ensure required folders exist

```powershell
mkdir templates -Force
mkdir models   -Force   # optional, not used by default
```

---

## Training the Model

The trained checkpoint is **not included** with the repository. You must train it once before starting the web app.

### Option A — Quick train (3 epochs, ~5 min on CPU, ~77% accuracy)

```powershell
python train_once.py
```

### Option B — Longer train (~30 epochs, ~30 min on CPU, ~87–90% accuracy)

Edit `train_once.py` and change `epochs=3` to `epochs=30`, then run:

```powershell
python train_once.py
```

This will:

1. Download CIFAR-10 into `./data/` (≈170 MB, first run only)
2. Train the CNN
3. Save the best checkpoint to `cifar10_cnn.pth` in the project root

You should see output like:

```
Using device: cpu
Epoch [1/3] - Test Loss: 1.0439, Test Acc: 62.40%
Saving model with accuracy: 62.40%
Epoch [2/3] - Test Loss: 0.9554, Test Acc: 67.70%
...
Training completed! Best accuracy: 77.59%
Saved cifar10_cnn.pth
```

---

## Running the Web App

```powershell
python flask_app.py
```

Expected output:

```
Model loaded successfully! Accuracy: 77.59%
 * Running on http://127.0.0.1:5000
```

Then open **http://127.0.0.1:5000** in your browser.

Upload a `.png` / `.jpg` / `.jpeg` image → click **Predict** → see the top-3 classes with confidence scores.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| POST | `/predict` | Upload image via form field `file`; returns JSON |
| GET | `/health` | Service status |

### Example `/predict` response

```json
{
  "predicted_class": "bird",
  "confidence": 0.9806,
  "top_3": [
    { "class": "bird",     "probability": 0.9806 },
    { "class": "airplane", "probability": 0.0091 },
    { "class": "frog",     "probability": 0.0043 }
  ]
}
```

### Example with curl

```powershell
curl -X POST -F "file=@test.jpg" http://127.0.0.1:5000/predict
```

---

## How It Works

1. **Preprocessing** (`image_preprocessing.py`): every uploaded image is resized to 32×32, converted to a tensor, and normalized with CIFAR-10 mean/std:
   ```
   mean = (0.4914, 0.4822, 0.4465)
   std  = (0.2023, 0.1994, 0.2010)
   ```
2. **Model** (`cifar10cnn.py`): three convolutional blocks (64→128→256 channels), each with two `Conv2d + BatchNorm + ReLU` layers followed by max-pooling. Then a fully-connected head with dropout.
3. **Inference** (`predict_image.py`): runs a forward pass, applies softmax, and returns the top-3 classes.
4. **Serving** (`flask_app.py`): receives uploaded bytes, calls `predict()`, and returns JSON.

---

## Tips for Best Results

CIFAR-10 images are **32×32 and low-resolution**. The model works best when:

- The **object fills most of the frame**
- The background is **simple / uniform**
- The image is a **clear, well-lit** shot of one object
- You avoid photos with multiple objects, heavy clutter, or extreme lighting

Real-world photos often yield lower confidence than the ~77–90% test accuracy suggests, because they look very different from the tiny CIFAR-10 thumbnails the model was trained on.

---

## Troubleshooting

### `Warning: Model file not found. Using untrained model.`

Run `python train_once.py` to create `cifar10_cnn.pth` in the project root. Confirm it exists:

```powershell
dir cifar10_cnn.pth
```

### `SSL: CERTIFICATE_VERIFY_FAILED` when downloading CIFAR-10

This is a Python/Windows certificate issue, not a code issue. Fixes:

- Reinstall Python from [python.org](https://www.python.org/downloads/) (avoid the Microsoft Store build)
- Or temporarily bypass in `train_once.py`:
  ```python
  import ssl
  ssl._create_default_https_context = ssl._create_unverified_context
  ```
- Or manually download `cifar-10-python.tar.gz` from [the CIFAR site](https://www.cs.toronto.edu/~kriz/cifar.html) and place it in `./data/`

### Template not found (`jinja2.exceptions.TemplateNotFound`)

Make sure the HTML file is at `templates/index.html`, not `flask_template.html` in the root.

### Predictions look random (e.g. 38% / 37% / 18%)

You are probably running an **untrained model**. Verify the checkpoint:

```powershell
python -c "import torch; ck=torch.load('cifar10_cnn.pth', map_location='cpu'); print('accuracy =', ck.get('accuracy'))"
```

If it prints `accuracy = 0.0` or a low value, retrain with `python train_once.py`.

---

## Tech Stack

- **PyTorch / TorchVision** — model and data loading
- **Flask** — web server and API
- **Pillow** — image decoding
- **NumPy** — numerical ops

---

## Notes

- `flask_app.py` runs Flask's **development server**. For production use, run it behind a WSGI server such as `waitress` (Windows) or `gunicorn` (Linux/macOS).
- `runtime.txt` pins Python 3.10.13 for platforms like Render/Heroku. Locally any Python 3.10–3.13 works.

---

## License

For educational use. CIFAR-10 dataset © Alex Krizhevsky, University of Toronto.