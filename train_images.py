"""Generic image classifier trainer (folder-per-class). Used for the urine microscopy model.
Usage: python train_images.py --data /content/urine_images --name urine --out /content/out
Classes are taken from sub-folder names. Any folder that directly holds >= 20 images counts as a class.
"""
import argparse, csv, glob, hashlib, json, os, random
from collections import defaultdict
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import classification_report, confusion_matrix

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def md5(p):
    with open(p, "rb") as f: return hashlib.md5(f.read()).hexdigest()

def collect(root, min_per_class=20):
    by_class = defaultdict(list)
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(EXTS):
            by_class[os.path.basename(os.path.dirname(p)).strip()].append(p)
    classes = sorted(c for c, v in by_class.items() if len(v) >= min_per_class)
    items, seen = [], set()
    for ci, c in enumerate(classes):
        for p in sorted(by_class[c]):
            h = md5(p)
            if h in seen: continue                 # drop exact duplicates (avoids train/test leakage)
            seen.add(h); items.append((p, ci))
    return classes, items

def split(items, n_classes):
    random.shuffle(items); out = {"train": [], "val": [], "test": []}
    for ci in range(n_classes):
        cls = [x for x in items if x[1] == ci]; n = len(cls)
        a, b = int(.7 * n), int(.85 * n)
        out["train"] += cls[:a]; out["val"] += cls[a:b]; out["test"] += cls[b:]
    return out

class DS(Dataset):
    def __init__(self, items, train, size):
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20), T.ColorJitter(.2, .2, .2)] if train else []
        self.tf = T.Compose([T.Resize((size, size)), *aug, T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]; return self.tf(Image.open(p).convert("RGB")), y

@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
    return ys, ps

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--name", default="urine")
    ap.add_argument("--out", default="out"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    classes, items = collect(a.data)
    print("classes:", classes); print("images per class:", {c: sum(1 for _, y in items if y == i) for i, c in enumerate(classes)})
    assert len(classes) >= 2, "Need at least 2 class folders with images. Check the data folder structure."
    sp = split(items, len(classes)); print({k: len(v) for k, v in sp.items()})
    with open(os.path.join(a.out, f"{a.name}_splits.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "split"])
        for k, v in sp.items():
            for p, y in v: w.writerow([os.path.relpath(p, a.data), classes[y], k])

    L = lambda k, t: DataLoader(DS(sp[k], t, a.size), batch_size=64, shuffle=t, num_workers=2)
    tr, va, te = L("train", True), L("val", False), L("test", False)
    counts = np.bincount([y for _, y in sp["train"]], minlength=len(classes)).astype(float)
    wts = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32).to(dev)  # handle imbalance

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes)); model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4); lossf = nn.CrossEntropyLoss(weight=wts)
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad(); lossf(model(x.to(dev)), y.to(dev)).backward(); opt.step()
        ys, ps = predict(model, va, dev)
        mf1 = classification_report(ys, ps, output_dict=True, zero_division=0)["macro avg"]["f1-score"]
        print(f"epoch {ep+1} val_macro_f1 {mf1:.4f}")
        if mf1 > best: best, best_state = mf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    ys, ps = predict(model, te, dev)   # test set used once, at the end
    names = classes
    rep = classification_report(ys, ps, labels=list(range(len(classes))), target_names=names, output_dict=True, zero_division=0)
    json.dump({"classes": classes, "img_size": a.size, "best_val_macro_f1": best, "test_report": rep,
               "confusion": confusion_matrix(ys, ps, labels=list(range(len(classes)))).tolist()},
              open(os.path.join(a.out, f"{a.name}_results.json"), "w"), indent=2)
    torch.save({"state_dict": best_state, "classes": classes, "img_size": a.size}, os.path.join(a.out, f"{a.name}_model.pt"))
    print(classification_report(ys, ps, labels=list(range(len(classes))), target_names=names, zero_division=0))
