# Urine & Parasite Detection

Small Flask app with two AI screening tools:
- **Parasite**: malaria blood-smear cell image classifier (MobileNetV2, 97.4% test accuracy on the NIH cell dataset, 4,134 test images).
- **Urine**: urine sediment microscopy particle classifier (MobileNetV2). Add `models/urine_model.pt` (made by `train_images.py`) and it activates automatically.

## Run locally
    pip install -r requirements.txt
    python app.py        # http://localhost:5000

## Deploy on Render
Build: `pip install -r requirements.txt`  Start: `gunicorn app:app`  (Python 3.12.8, see `render.yaml`).

## Data and training
Parasite: Kaggle `iarunava/cell-images-for-detecting-malaria`. Urine: see training submission.
Training scripts and split files are shipped separately in the training submission.
