"""Image classifiers (parasite + urine microscopy). Models are PyTorch MobileNetV2 checkpoints in models/."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))

def _path(name):
    """model file: next to app.py, or in a models/ folder"""
    for d in (_HERE, os.path.join(_HERE, "models")):
        p = os.path.join(d, f"{name}_model.pt")
        if os.path.exists(p): return p
    return None
_cache = {}

class ModelMissing(Exception):
    pass

def model_available(name):
    return _path(name) is not None

def _load(name):
    if name not in _cache:
        path = _path(name)
        if not path:
            raise ModelMissing(f"The {name} model is not installed yet.")
        import torch, torch.nn as nn
        from torchvision.models import mobilenet_v2
        ckpt = torch.load(path, map_location="cpu")
        m = mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, len(ckpt["classes"]))
        m.load_state_dict(ckpt["state_dict"]); m.eval()
        _cache[name] = (m, ckpt["classes"], int(ckpt.get("img_size", 128)))
    return _cache[name]

def predict_image(name, img):
    if not model_available(name):
        raise ModelMissing(f"The {name} model is not installed yet.")
    import torch
    import torchvision.transforms as T
    model, classes, size = _load(name)
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    with torch.no_grad():
        p = torch.softmax(model(tf(img.convert("RGB")).unsqueeze(0)), 1)[0]
    order = sorted(range(len(classes)), key=lambda j: -float(p[j]))
    return {"label": classes[order[0]], "confidence": round(float(p[order[0]]), 4),
            "probabilities": {classes[j]: round(float(p[j]), 4) for j in order}}
