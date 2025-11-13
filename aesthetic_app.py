from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
import numpy as np
import tempfile
import os

# ------------ Configuration ------------
BTN_BG = "#003366"

COLOR_MAP = {
    "Acne": (0, 0, 255),
    "Blackheads": (0, 255, 0),
    "Dark Circles": (255, 0, 0),
    "Pigmentation": (0, 255, 255),
    "Wrinkles": (255, 0, 255),
    "Pore": (255, 255, 0),
    "Redness": (255, 165, 0),
}
THRESHOLDS = {
    "Acne": 0.6,
    "Blackheads": 0.5,
    "Dark Circles": 0.5,
    "Pigmentation": 0.5,
    "Wrinkles": 0.4,
    "Pore": 0.6,
    "Redness": 0.6,
}
DROP = {
    "Acne": 9,
    "Blackheads": 8,
    "Dark Circles": 14,
    "Pigmentation": 23,
    "Wrinkles": 18,
    "Pore": 17,
    "Redness": 27,
}
EXPLANATIONS = {
    "Acne": "Acne is caused by clogged pores, bacteria, and excess oil production.",
    "Blackheads": "Blackheads form when pores are partially clogged with oil and dead skin cells.",
    "Dark Circles": "Dark circles can be due to genetics, thin under-eye skin, or hyperpigmentation.",
    "Pigmentation": "Pigmentation irregularities arise from melanin overproduction, triggered by sun exposure.",
    "Wrinkles": "Wrinkles develop from loss of collagen and elastin over time.",
    "Pore": "Enlarged pores result from genetics, oiliness, and loss of skin elasticity.",
    "Redness": "Redness can be caused by irritation, inflammation, or vascular issues.",
}
SUGGESTIONS = {
    "Acne": "Use a gentle salicylic acid wash and consult a dermatologist.",
    "Blackheads": "Try chemical exfoliation with BHA (salicylic acid).",
    "Dark Circles": "Ensure sufficient sleep and use cold compresses.",
    "Pigmentation": "Apply daily SPF 30+ and use topical lightening agents.",
    "Wrinkles": "Incorporate retinoids at night and protect skin from UV.",
    "Pore": "Use clay masks weekly and oil-control primers.",
    "Redness": "Choose fragrance-free, soothing products.",
}

def quality_label(pct: float) -> str:
    if pct > 80:
        return "Good"
    elif pct >= 40:
        return "Neutral"
    else:
        return "Poor"

# ------------ App init ------------
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# static + templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# PDF font
font_path = BASE_DIR / "Fonts" / "League_Spartan" / "static" / "LeagueSpartan-SemiBold.ttf"
if font_path.exists():
    pdfmetrics.registerFont(TTFont("LeagueSpartan", str(font_path)))

# YOLO modellerini yükle
models_dir = BASE_DIR / "Modeller" / "Medical Aesthetic"
file_names = {
    "Acne": "acne.pt",
    "Blackheads": "Blackheads.pt",
    "Dark Circles": "Dark Circles.pt",
    "Pigmentation": "Pigmentation.pt",
    "Wrinkles": "Wrinkles.pt",
    "Pore": "pore.pt",
    "Redness": "redness.pt",
}
MODELS = {name: YOLO(str(models_dir / fname)) for name, fname in file_names.items()}

# ------------ Routes ------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "error": None},
    )

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": "Image could not be read."},
        )

    annotated = frame.copy()
    report_data = []

    for name, model in MODELS.items():
        results = model(frame, conf=THRESHOLDS[name])[0]
        count = len(results.boxes)
        pct = max(0, 100 - DROP[name] * count)
        pct = max(0, min(100, pct))

        color = COLOR_MAP[name]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        quality = quality_label(pct)
        chip_class = (
            "good" if quality == "Good"
            else "neutral" if quality == "Neutral"
            else "poor"
        )

        report_data.append(
            {
                "name": name,
                "quality": quality,
                "chip_class": chip_class,
                "score_pct": int(pct),
                "explanation": EXPLANATIONS[name],
                "suggestion": SUGGESTIONS[name],
            }
        )

    # Annotated image kaydet
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotated_name = f"annotated_{ts}.png"
    annotated_path = RESULTS_DIR / annotated_name
    cv2.imwrite(str(annotated_path), annotated)

    # PDF üret
    pdf_name = f"Report_{ts}.pdf"
    pdf_path = RESULTS_DIR / pdf_name
    generate_pdf(annotated_path, pdf_path, report_data)

    result = {
        "annotated_url": f"/static/results/{annotated_name}",
        "pdf_url": f"/static/results/{pdf_name}",
        "items": report_data,
    }

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "error": None},
    )

def generate_pdf(image_path: Path, pdf_path: Path, data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()

    img = cv2.imread(str(image_path))
    cv2.imwrite(tmp.name, img)

    c = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4
    img_w, img_h = 400, 300
    x, y = (w - img_w) / 2, h - img_h - 50

    c.drawImage(ImageReader(tmp.name), x, y, img_w, img_h)

    text_y = y - 30
    for item in data:
        name = item["name"]
        quality = item["quality"]
        expl = item["explanation"]
        sugg = item["suggestion"]

        font_name = (
            "LeagueSpartan"
            if "LeagueSpartan" in pdfmetrics.getRegisteredFontNames()
            else "Helvetica"
        )

        c.setFont(font_name, 12)
        c.setFillColor(HexColor("#000000"))
        c.drawString(50, text_y, f"{name} - {quality}")
        text_y -= 20

        frac = {"Good": 1.0, "Neutral": 0.6, "Poor": 0.2}[quality]
        bar_len = frac * (w - 100)
        c.setFillColor(HexColor(BTN_BG))
        c.rect(50, text_y - 10, bar_len, 15, fill=1, stroke=0)
        text_y -= 30

        c.setFillColor(HexColor("#000000"))
        c.setFont(font_name, 10)
        c.drawString(50, text_y, f"Explanation: {expl}")
        text_y -= 18
        c.drawString(50, text_y, f"Suggestion: {sugg}")
        text_y -= 30

        if text_y < 80:
            c.showPage()
            c.drawImage(ImageReader(tmp.name), x, y, img_w, img_h)
            text_y = y - 30

    c.save()
    os.unlink(tmp.name)
