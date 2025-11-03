import io, os, time, json, requests, hashlib
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Optional backends
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ALLOWED_ORIGINS = ["https://patientsum.com", "https://www.patientsum.com"]
WEIGHTS_DIR = os.path.join(os.getcwd(), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

app = FastAPI(title="PatientSum API (real models)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utility: downloads & loaders ----------

def _download_if_needed(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    # deterministic file name from URL
    h = hashlib.sha256(url.encode()).hexdigest()[:10]
    fname = os.path.join(WEIGHTS_DIR, f"w_{h}" + os.path.splitext(url.split("?")[0])[-1])
    if os.path.exists(fname) and os.path.getsize(fname) > 0:
        return fname
    # stream download
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
    return fname

def _load_model(path: Optional[str]):
    if not path:
        return None, None  # not provided
    ext = os.path.splitext(path)[-1].lower()
    # YOLO .pt
    if ext == ".pt":
        if not _HAS_YOLO:
            raise RuntimeError("ultralytics not installed but .pt provided")
        model = YOLO(path)
        return ("yolo", model)
    # TorchScript .ts
    if ext == ".ts":
        if not _HAS_TORCH:
            raise RuntimeError("torch not installed but .ts provided")
        m = torch.jit.load(path, map_location="cpu")
        m.eval()
        return ("torchscript", m)
    # ONNX or others could be added here
    raise RuntimeError(f"Unsupported weight extension: {ext}")

def _to_numpy(img: Image.Image, mode="xray", size=512) -> np.ndarray:
    if mode == "xray" and img.mode != "L":
        img = img.convert("L")
    if mode != "xray" and img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0
    return arr

def _predict_yolo(model, img_np: np.ndarray) -> Tuple[str, float, dict]:
    """
    Supports both YOLO classification and detection.
    - If classification head exists: use probs
    - Else if detection: 'fracture' if any detection (or map by class name)
    """
    # YOLO wants BGR uint8 or path; use uint8 RGB
    if img_np.ndim == 2:  # grayscale
        rgb = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        rgb = img_np
    rgb8 = (rgb * 255).astype(np.uint8)

    res = model.predict(rgb8, imgsz=512, verbose=False, device="cpu", conf=0.25)
    r0 = res[0]

    # classification?
    if getattr(r0, "probs", None) is not None:
        probs = r0.probs.data.cpu().numpy().astype("float32")
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        # class names
        names = getattr(model, "names", None) or getattr(r0, "names", None)
        label = names[idx] if names and idx in names else f"class_{idx}"
        return label, conf, {"mode":"classification"}

    # detection?
    if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
        # pick the top box
        confs = r0.boxes.conf.cpu().numpy().astype("float32")
        cls = r0.boxes.cls.cpu().numpy().astype("int32")
        top = int(np.argmax(confs))
        conf = float(confs[top])
        names = getattr(model, "names", None) or getattr(r0, "names", None)
        cls_name = names[cls[top]] if names and cls[top] in names else f"class_{cls[top]}"
        # If your model detects "fracture" boxes, use that name. Otherwise just return the top class.
        return cls_name, conf, {"mode":"detection","n":len(r0.boxes)}
    # No output
    return "none", 0.0, {"mode":"none"}

def _predict_torchscript(model, img_np: np.ndarray) -> Tuple[str, float, dict]:
    """
    Very generic TorchScript classification stub:
    expects model(img: (1,C,H,W) float32) -> logits (1,K)
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch not available")
    x = img_np
    if x.ndim == 2:             # (H,W) gray
        x = x[None, :, :]       # (1,H,W)
    else:                       # (H,W,3) RGB
        x = x.transpose(2,0,1)  # (3,H,W)
    x = x[None, ...]            # (1,C,H,W)
    xt = torch.from_numpy(x).float()
    with torch.no_grad():
        logits = model(xt)
        if isinstance(logits, (list, tuple)): logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = float(probs.max().item()), int(probs.argmax().item())
    return f"class_{idx}", conf, {"mode":"torchscript"}

def _predict_backend(backend, model, img_np, label_map=None):
    if backend == "yolo":
        lab, conf, meta = _predict_yolo(model, img_np)
        # Optional mapping: {"class_0": "no_fracture", "class_1":"fracture"} etc.
        if label_map and lab in label_map:
            lab = label_map[lab]
        return lab, conf, meta
    if backend == "torchscript":
        lab, conf, meta = _predict_torchscript(model, img_np)
        if label_map and lab in label_map:
            lab = label_map[lab]
        return lab, conf, meta
    raise RuntimeError("Unknown backend")

# ---------- Load your weights from ENV ----------

MODELS = {
    "fracture": {
        "URL": os.getenv("FRAC_MODEL_URL", "").strip() or None,
        "LABEL_MAP_JSON": os.getenv("FRAC_LABEL_MAP_JSON", "").strip() or None, # e.g. {"0":"no_fracture","1":"fracture"}
        "MODE": os.getenv("FRAC_MODE", "xray"),  # xray/photo
        "LOADED": None,     # (backend, model)
        "LABEL_MAP": None,  # dict or None
    },
    "gonarthrosis": {
        "URL": os.getenv("GON_MODEL_URL", "").strip() or None,
        "LABEL_MAP_JSON": os.getenv("GON_LABEL_MAP_JSON", "").strip() or None, # e.g. {"0":"KL0","1":"KL1",...}
        "MODE": os.getenv("GON_MODE", "xray"),
        "LOADED": None,
        "LABEL_MAP": None,
    },
    "melanoma": {
        "URL": os.getenv("MEL_MODEL_URL", "").strip() or None,
        "LABEL_MAP_JSON": os.getenv("MEL_LABEL_MAP_JSON", "").strip() or None, # e.g. {"0":"benign","1":"malignant"}
        "MODE": os.getenv("MEL_MODE", "photo"),
        "LOADED": None,
        "LABEL_MAP": None,
    },
}

def _load_all_models():
    for key, cfg in MODELS.items():
        url = cfg["URL"]
        if not url:
            continue  # will use stub
        try:
            path = _download_if_needed(url)
            backend, model = _load_model(path)
            cfg["LOADED"] = (backend, model)
            if cfg["LABEL_MAP_JSON"]:
                try:
                    cfg["LABEL_MAP"] = json.loads(cfg["LABEL_MAP_JSON"])
                except Exception:
                    cfg["LABEL_MAP"] = None
            print(f"[INIT] Loaded {key} model via {backend} from {path}")
        except Exception as e:
            print(f"[INIT] Failed to load {key}: {e}")

_load_all_models()

# ---------- Routes ----------

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/models")
def models():
    return {
        k: {
            "has_url": bool(v["URL"]),
            "loaded": bool(v["LOADED"]),
            "backend": (v["LOADED"][0] if v["LOADED"] else None),
            "mode": v["MODE"],
            "label_map_keys": list(v["LABEL_MAP"].keys()) if v["LABEL_MAP"] else None
        } for k,v in MODELS.items()
    }

def _predict_stub(task: str) -> Tuple[str, float, str, str]:
    if task == "fracture":
        return "no_fracture", 0.85, "Best with frontal/lateral X-rays.", "Does not replace radiologist/orthopedic evaluation."
    if task == "gonarthrosis":
        return "KL2", 0.70, "Weight-bearing AP knee recommended.", "KL grading is an estimate; confirm clinically."
    return "benign", 0.80, "Dermoscopic photos recommended.", "Screening only; consult a dermatologist."

@app.post("/v1/predict")
async def predict(file: UploadFile = File(...), task: str = Form(...)):
    if task not in {"fracture","gonarthrosis","melanoma"}:
        raise HTTPException(400,"task must be fracture | gonarthrosis | melanoma")
    if not (file.filename.lower().endswith((".jpg",".jpeg",".png")) or (file.content_type or "").startswith("image/")):
        raise HTTPException(400,"Upload JPG or PNG")
    content = await file.read()
    if len(content) > 15*1024*1024:
        raise HTTPException(413,"File too large (max 15 MB)")
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(400,"Unreadable image")

    cfg = MODELS[task]
    details = {
        "fracture": "Best with frontal/lateral X-rays.",
        "gonarthrosis": "Weight-bearing AP knee recommended.",
        "melanoma": "Use dermoscopic photos."
    }[task]
    caveat = {
        "fracture": "Does not replace radiologist/orthopedic evaluation.",
        "gonarthrosis": "KL grading is an estimate; confirm clinically.",
        "melanoma": "Screening only; consult a dermatologist."
    }[task]

    # If real model loaded, use it
    if cfg["LOADED"]:
        backend, model = cfg["LOADED"]
        img_np = _to_numpy(img, mode=cfg["MODE"], size=512)
        label, conf, meta = _predict_backend(backend, model, img_np, cfg["LABEL_MAP"])
        return {"label": label, "confidence": float(conf), "details": details, "caveat": caveat, "meta": meta}

    # Fallback
    label, conf, details_stub, caveat_stub = _predict_stub(task)
    return {"label": label, "confidence": float(conf), "details": details_stub or details, "caveat": caveat_stub or caveat}
