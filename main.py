import io, numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PatientSum API")

ALLOWED_ORIGINS = ["https://patientsum.com", "https://www.patientsum.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def predict_stub(labels):
    probs = np.array([0.85] + [0.15/(len(labels)-1)]*(len(labels)-1))
    i = int(probs.argmax()); return labels[i], float(probs.max())

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/v1/predict")
async def predict(file: UploadFile = File(...), task: str = Form(...)):
    if task not in {"fracture","gonarthrosis","melanoma"}:
        raise HTTPException(400,"task must be fracture | gonarthrosis | melanoma")
    if not (file.filename.lower().endswith((".jpg",".jpeg",".png")) or (file.content_type or "").startswith("image/")):
        raise HTTPException(400,"Upload JPG or PNG")

    content = await file.read()
    if len(content) > 15*1024*1024: raise HTTPException(413,"File too large (max 15 MB)")
    try: Image.open(io.BytesIO(content))
    except Exception: raise HTTPException(400,"Unreadable image")

    if task=="fracture":
        label, conf = predict_stub(["no_fracture","fracture"])
        details = "Best with frontal/lateral X-rays."
        caveat = "Does not replace radiologist/orthopedic evaluation."
    elif task=="gonarthrosis":
        label, conf = predict_stub(["KL0","KL1","KL2","KL3","KL4"])
        details = "Weight-bearing AP knee recommended."
        caveat = "Estimate; confirm clinically."
    else:
        label, conf = predict_stub(["benign","malignant"])
        details = "Use dermoscopic photos."
        caveat = "Screening only; consult a dermatologist."

    return {"label":label, "confidence":conf, "details":details, "caveat":caveat}
