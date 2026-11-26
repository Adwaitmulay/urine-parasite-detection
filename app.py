"""Urine & Parasite Detection: small standalone Flask app.
Run locally: python app.py     Render: gunicorn app:app
"""
import io, os
from flask import Flask, jsonify, render_template, request
from PIL import Image
import inference

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=HERE)   # templates live next to app.py (flat repo layout)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024   # 8 MB upload limit
ALLOWED = {"png", "jpg", "jpeg", "jfif", "bmp", "tif", "tiff", "webp"}

@app.route("/")
def index(): return render_template("index.html", page="home", urine_ready=inference.model_available("urine"))

@app.route("/parasite")
def parasite_page(): return render_template("parasite.html", page="parasite")

@app.route("/urine")
def urine_page(): return render_template("urine.html", page="urine", ready=inference.model_available("urine"))

@app.route("/healthz")
def health(): return "ok"

def _predict(name):
    f = request.files.get("image")
    if not f or not f.filename or f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return jsonify(success=False, error="Please upload a valid image (JPG, PNG...)."), 400
    try:
        img = Image.open(io.BytesIO(f.read())); img.load()
        return jsonify(success=True, **inference.predict_image(name, img))
    except inference.ModelMissing as e:
        return jsonify(success=False, error=str(e)), 503
    except Exception as e:
        return jsonify(success=False, error=f"Could not process image: {e}"), 500

@app.route("/predict_parasite", methods=["POST"])
def predict_parasite(): return _predict("parasite")

@app.route("/predict_urine", methods=["POST"])
def predict_urine(): return _predict("urine")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
