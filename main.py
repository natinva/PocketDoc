import os
import io
import base64
import hashlib
import time
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import requests
import cv2

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aesthetic_routes import router as aesthetic_router

app = FastAPI()

# static mount yoksa ekle
app.mount("/static", StaticFiles(directory="static"), name="static")

# mevcut kendi router’ların vs burada...

# aesthetic modülünü ekle
app.include_router(aesthetic_router)

# ----------------------------
# Config & Model URLs
# ----------------------------
API_ALLOWED_ORIGINS = [
    "https://patientsum.com",
    "https://www.patientsum.com",
    "https://cepdoktorum.com",
    "https://www.cepdoktorum.com",
    "http://localhost:5173",
    "http://localhost:5500",
]

# You can set these on Render → Environment
FRACTURE_URL = os.getenv(
    "MODEL_FRACTURE_URL",
    "https://github.com/natinva/PocketDoc/releases/download/Models/fracture.pt",
)
KNEE_URL = os.getenv(
    "MODEL_KNEE_URL",
    "https://github.com/natinva/PocketDoc/releases/download/Models/knee.pt",
)

# If you bring melanoma back later, you can add it here
MELANOMA_URL = os.getenv(
    "MODEL_MELANOMA_URL",
    "https://github.com/natinva/PocketDoc/releases/download/Models/melanom.pt",
)

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# App init
# ----------------------------
app = FastAPI(title="PatientSum Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Utilities
# ----------------------------
def _fname_from_url(url: str) -> str:
    # stable filename even if url has query
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    base = url.split("/")[-1] or f"model-{h}.pt"
    if not base.endswith(".pt"):
        base = base + ".pt"
    return f"{h}-{base}"

def _download_once(url: str) -> str:
    """Download to MODEL_DIR if missing; return local path."""
    if not url:
        raise RuntimeError("Empty model URL")
    fname = _fname_from_url(url)
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    # stream download
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return path

def _to_data_url_png(img_rgb: np.ndarray) -> str:
    """img_rgb expected in RGB uint8, return data:image/png;base64,..."""
    if img_rgb is None:
        return ""
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    # cv2 wants BGR
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

# ----------------------------
# Load Ultralytics YOLO lazily
# ----------------------------
YOLO = None
MODELS: Dict[str, Any] = {
    "fracture": None,
    "gonarthrosis": None,  # KL 0-4
    # "melanoma": None,    # add back later if needed
}
MODEL_URLS: Dict[str, Optional[str]] = {
    "fracture": FRACTURE_URL,
    "gonarthrosis": KNEE_URL,
    # "melanoma": MELANOMA_URL,
}

def load_yolo():
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO

def load_model(task: str):
    """Load a YOLO model for a given task if not loaded."""
    if task not in MODELS:
        raise RuntimeError(f"Unknown task: {task}")
    if MODELS[task] is not None:
        return MODELS[task]
    load_yolo()
    url = MODEL_URLS.get(task)
    has_url = bool(url)
    if has_url:
        local = _download_once(url)
        model = YOLO(local)
    else:
        raise RuntimeError(f"No URL for model '{task}'")
    MODELS[task] = model
    return model

def warmup_models():
    """Optional: warmup both models once to reduce first-pred latency."""
    try:
        for t in ["fracture", "gonarthrosis"]:
            m = load_model(t)
            # warmup with tiny white image
            dummy = np.full((320, 320, 3), 255, dtype=np.uint8)
            _ = m(dummy)
    except Exception:
        # warmup is best-effort; don't crash startup
        pass

# Warm up in background-ish manner
try:
    warmup_models()
except Exception:
    pass

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": int(time.time())}

@app.get("/models")
def models_status():
    out = {}
    for name in ["fracture", "gonarthrosis", "melanoma"]:
        url = MODEL_URLS.get(name)
        loaded = MODELS.get(name) is not None
        if name == "melanoma":
            # currently disabled (per your last request to remove)
            out[name] = {
                "has_url": bool(url),
                "loaded": False,
                "backend": None,
                "mode": "photo",
                "label_map_keys": None,
            }
        else:
            out[name] = {
                "has_url": bool(url),
                "loaded": loaded,
                "backend": "yolo" if loaded else None,
                "mode": "xray" if name in ("fracture", "gonarthrosis") else "photo",
                "label_map_keys": ["0", "1"] if name == "fracture" else ["0", "1", "2", "3", "4"],
            }
    return out

@app.post("/v1/predict")
async def predict(file: UploadFile = File(...), task: str = Form(...)):
    task = task.strip().lower()
    if task not in MODELS:
        return JSONResponse({"error": f"Unsupported task '{task}'"}, status_code=400)

    # Read image
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"Invalid image: {e}"}, status_code=400)
    img_np = np.array(img)

    # Load model (lazy) & run
    try:
        model = load_model(task)
        results = model(img_np)  # Ultralytics API
        res = results[0]
    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {e}"}, status_code=500)

    # Build annotated image
    try:
        plotted = res.plot()  # Ultralytics returns RGB np.ndarray
        # Some builds may return BGR; detect by a quick heuristic if needed
        # We'll trust it's RGB; PNG encode downstream handles conversion
        annotated_data_url = _to_data_url_png(plotted)
        annotated_ok = True
    except Exception:
        # fallback: original
        annotated_data_url = _to_data_url_png(img_np)
        annotated_ok = False

    # Extract prediction summary
    label = "No finding"
    conf = 0.0
    details = None
    caveat = None

    # 1) If classification probs exist (rare for your use), prefer them
    if getattr(res, "probs", None) is not None and res.probs is not None:
        # res.probs.top1, res.names dict, res.probs.data (Tensor)
        top1 = int(res.probs.top1)
        label = res.names.get(top1, f"class_{top1}")
        try:
            conf = float(res.probs.top1conf)
        except Exception:
            conf = float(res.probs.data[top1]) if hasattr(res.probs, "data") else 0.0
    else:
        # 2) Use boxes (detection). If no boxes, default "No finding"
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # pick highest conf box
            confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.array(boxes.conf)
            idx = int(np.argmax(confs))
            cls_id = int(boxes.cls[idx])
            names = getattr(res, "names", {}) or {}
            raw_label = names.get(cls_id, f"class_{cls_id}")
            if task == "gonarthrosis":
                # map to KL wording
                # Expect classes 0..4 -> KL 0..4
                label = f"KL grade {cls_id}"
                details = "Automated Kellgren–Lawrence estimate; confirm clinically."
            else:
                # fracture: assume 1=fracture, 0=no fracture in your training
                label = raw_label
            conf = float(confs[idx])

    # Clinical notes (fixed brief)
    if task == "gonarthrosis":
        details = (
            "Kellgren–Lawrence (KL) grading estimates radiographic OA severity from 0 (none) to 4 (severe). "
            "Features include osteophytes, joint-space narrowing, subchondral sclerosis and bony deformity. "
            "This is an automated estimate; correlate with clinical exam."
        )
    elif task == "fracture":
        details = (
            "AI fracture screening highlights suspicious regions; absence of a box does not exclude fracture. "
            "Projection, positioning and artifacts can affect results. Consider clinical exam and follow-up imaging."
        )

    # If we failed to annotate, set caveat
    if not annotated_ok:
        caveat = "Showing original image (annotation unavailable)."

    payload = {
        "label": label,
        "confidence": conf,
        "details": details,
        "caveat": caveat,
        "annotated_image": annotated_data_url,  # <-- frontend displays this
    }
    return JSONResponse(payload)
