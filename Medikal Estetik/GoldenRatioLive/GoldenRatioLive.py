import cv2
import mediapipe as mp
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter.messagebox as messagebox
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import time
import io
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─── find your repo root (one level above "Medikal Estetik") ──────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # .../PocketDoc/Medikal Estetik
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))       # .../PocketDoc

FONT_PATH = os.path.join(
    REPO_ROOT,
    "Fonts", "League_Spartan", "static",
    "LeagueSpartan-SemiBold.ttf"
)
# optional sanity check:
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Cannot find font at {FONT_PATH!r}")

pdfmetrics.registerFont(
    TTFont("LeagueSpartan-SemiBold", FONT_PATH)
)


def export_pdf(pil_img, report_text, filename="report.pdf"):
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from tkinter import messagebox

    # Save the PIL image into a bytes buffer
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    img_reader = ImageReader(buf)

    # Create the PDF canvas
    c = canvas.Canvas(filename, pagesize=letter)
    pw, ph = letter

    # Draw the image on the left half of the page
    iw, ih = pil_img.size
    max_w, max_h = pw/2 - 40, ph - 80
    scale = min(max_w/iw, max_h/ih)
    c.drawImage(img_reader,
                x=20,
                y=ph - ih*scale - 40,
                width=iw*scale,
                height=ih*scale)

    # Draw the report text on the right half, using your Turkish-capable font
    c.setFont("LeagueSpartan-SemiBold", 10)
    text = c.beginText(pw/2 + 20, ph - 40)
    text.setLeading(14)
    for line in report_text.splitlines():
        text.textLine(line)
    c.drawText(text)

    c.save()
    messagebox.showinfo("PDF Saved", f"Report saved to {filename}")

# ─── Set up MediaPipe FaceMesh ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # video mode
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ─── Landmark Indices ───────────────────────────────────────────────────────
bottom_nose_idx         = 2
top_lip_idx             = 13
bottom_lip_idx          = 14
chin_idx                = 152
left_eye_inner_idx      = 133
right_eye_inner_idx     = 362
glabella_idx            = 168
top_forehead_idx        = 10
leftmost_face_idx       = 234
rightmost_face_idx      = 454
nose_tip_idx            = 1
chin_tip_idx            = 152
left_inner_eyebrow_idx  = 55
right_inner_eyebrow_idx = 285
left_eye_center_idx     = 468
right_eye_center_idx    = 473
left_eye_outer_idx      = 33
right_eye_outer_idx     = 263
left_mouth_corner_idx   = 61
right_mouth_corner_idx  = 291
left_nose_wing_idx      = 49
right_nose_wing_idx     = 279

# Ideal (normal) values for each ratio
IDEAL_VALUES = {
    "Burun-Dudak-Çene Oranı":       1.618,  # Nose-Lips-Chin
    "Göz Simetrisi Oranı":          1.0,    # Eye Symmetry
    "Yüz H/W Oranı":                1.618,  # Face Height/Width
    "Burun-Çene/Alın Oranı":        1.618,  # C/D
    "Göz Bebeği/Genislik Oranı":     1.0,    # Pupil/Width
    "Ağız/Burun Oranı":             1.618,  # Mouth/Nose
    "Çene-Yüz Yükseklik-Genişlik":   2.0     # Jawline / Face Width
}

# Mini‐info sentences for each ratio
RATIO_INFO = {
    "Burun-Dudak-Çene Oranı":      "Bu oran, burunla üst dudak ve alt dudakla çene arasındaki mesafelerin oranını gösterir.",
    "Göz Simetrisi Oranı":         "Bu oran, her iki gözün glabella noktasına eşit mesafede olup olmadığını ölçer.",
    "Yüz H/W Oranı":               "Yüzün alın-çene yüksekliği ile yüzün yanak genişliği arasındaki oranı temsil eder.",
    "Burun-Çene/Alın Oranı":       "Bu oran, burun ucu ile çene ucu arasının, alın yüksekliğine oranını tanımlar.",
    "Göz Bebeği/Genislik Oranı":    "Gözbebekleri arasındaki mesafe ile her bir gözün genişliği arasındaki oranı hesaplar.",
    "Ağız/Burun Oranı":            "Ağız açıklığı genişliği ile burun genişliği arasındaki orandır.",
    "Çene-Yüz Yükseklik-Genişlik":  "Çene hattı boyunca uzanan mesafe ile yüzün sağ-sol genişliği arasındaki orandır."
}

# ─── Helper: Find “highest forehead” with 10% offset ────────────────────────
def calculate_highest_point_with_percentage_offset(face_landmarks, img_h, img_w):
    top_forehead = face_landmarks.landmark[top_forehead_idx]
    chin = face_landmarks.landmark[chin_idx]

    tx = int(top_forehead.x * img_w)
    ty = int(top_forehead.y * img_h)
    cy = int(chin.y * img_h)

    face_h = cy - ty
    offset = int(face_h * 0.1)  # 10% above top_forehead
    highest_y = max(0, ty - offset)
    return tx, highest_y

# ─── “draw‐on‐frame” routines (unchanged from before, minus commented upload) ─
def show_nose_lips_chin_ratio_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        bn_x, bn_y = int(lm[bottom_nose_idx].x * w),   int(lm[bottom_nose_idx].y * h)
        tl_x, tl_y = int(lm[top_lip_idx].x * w),       int(lm[top_lip_idx].y * h)
        bl_x, bl_y = int(lm[bottom_lip_idx].x * w),    int(lm[bottom_lip_idx].y * h)
        ch_x, ch_y = int(lm[chin_idx].x * w),          int(lm[chin_idx].y * h)

        d1 = math.hypot(bl_x - ch_x, bl_y - ch_y)
        d2 = math.hypot(bn_x - tl_x, bn_y - tl_y)
        ratio = d1 / d2 if d2 else 0

        cv2.line(img, (bn_x, bn_y), (tl_x, tl_y), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (bl_x, bl_y), (ch_x, ch_y), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Nose-Lips-Chin: {ratio:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: 1.618", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def show_eye_symmetry_ratio_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        lx, ly = int(lm[left_eye_inner_idx].x * w),  int(lm[left_eye_inner_idx].y * h)
        rx, ry = int(lm[right_eye_inner_idx].x * w), int(lm[right_eye_inner_idx].y * h)
        gx, gy = int(lm[glabella_idx].x * w),       int(lm[glabella_idx].y * h)

        d_l = math.hypot(lx - gx, ly - gy)
        d_r = math.hypot(rx - gx, ry - gy)
        ratio = d_l / d_r if d_r else 0

        cv2.line(img, (lx, ly), (gx, gy), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, (rx, ry), (gx, gy), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Eye Symmetry: {ratio:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: 1.00", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def show_both_ratios_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        bn_x, bn_y = int(lm[bottom_nose_idx].x * w), int(lm[bottom_nose_idx].y * h)
        tl_x, tl_y = int(lm[top_lip_idx].x * w),       int(lm[top_lip_idx].y * h)
        bl_x, bl_y = int(lm[bottom_lip_idx].x * w),    int(lm[bottom_lip_idx].y * h)
        ch_x, ch_y = int(lm[chin_idx].x * w),          int(lm[chin_idx].y * h)
        d_nlc = math.hypot(bl_x - ch_x, bl_y - ch_y) / (math.hypot(bn_x - tl_x, bn_y - tl_y) or 1)

        lx, ly = int(lm[left_eye_inner_idx].x * w),  int(lm[left_eye_inner_idx].y * h)
        rx, ry = int(lm[right_eye_inner_idx].x * w), int(lm[right_eye_inner_idx].y * h)
        gx, gy = int(lm[glabella_idx].x * w),       int(lm[glabella_idx].y * h)
        d_eye = math.hypot(lx - gx, ly - gy) / (math.hypot(rx - gx, ry - gy) or 1)

        cv2.line(img, (bn_x, bn_y), (tl_x, tl_y), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (bl_x, bl_y), (ch_x, ch_y), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (lx, ly),   (gx, gy),     (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, (rx, ry),   (gx, gy),     (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(img, f"N-L-C: {d_nlc:.2f} (1.618)", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Eye Sym: {d_eye:.2f} (1.00)", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def show_face_height_to_width_ratio_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        flm = results.multi_face_landmarks[0]
        lm = flm.landmark
        highest_x, highest_y = calculate_highest_point_with_percentage_offset(flm, h, w)

        chin = lm[chin_idx]
        ch_x, ch_y = int(chin.x * w), int(chin.y * h)

        left  = lm[leftmost_face_idx]
        right = lm[rightmost_face_idx]
        l_x, l_y   = int(left.x * w),  int(left.y * h)
        r_x, r_y   = int(right.x * w), int(right.y * h)

        dh = math.hypot(highest_x - ch_x, highest_y - ch_y)
        dw = math.hypot(l_x - r_x,     l_y - r_y)
        ratio = dh / dw if dw else 0

        cv2.line(img, (highest_x, highest_y), (ch_x, ch_y), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (l_x, l_y),         (r_x, r_y),     (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(img, f"Face H/W: {ratio:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: 1.618", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return img

def show_cd_ratio_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        flm = results.multi_face_landmarks[0]
        lm = flm.landmark
        highest_x, highest_y = calculate_highest_point_with_percentage_offset(flm, h, w)

        nose = lm[nose_tip_idx]
        chin = lm[chin_tip_idx]
        nx, ny = int(nose.x * w), int(nose.y * h)
        cx, cy = int(chin.x * w), int(chin.y * h)
        C = math.hypot(nx - cx, ny - cy)

        le = lm[left_inner_eyebrow_idx]
        re = lm[right_inner_eyebrow_idx]
        lx, ly = int(le.x * w), int(le.y * h)
        rx, ry = int(re.x * w), int(re.y * h)
        mid_x, mid_y = (lx + rx)//2, (ly + ry)//2

        D = math.hypot(highest_x - mid_x, highest_y - mid_y)
        ratio = C / D if D else 0

        cv2.line(img, (nx, ny),         (cx, cy),         (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (highest_x, highest_y), (mid_x, mid_y), (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(img, f"Nose-Chin/Forehd: {ratio:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: 1.618", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def show_eye_pupil_to_width_ratio_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        lc_x, lc_y = lm[left_eye_center_idx].x * w,  lm[left_eye_center_idx].y * h
        rc_x, rc_y = lm[right_eye_center_idx].x * w, lm[right_eye_center_idx].y * h
        li_x, li_y = lm[left_eye_inner_idx].x * w,  lm[left_eye_inner_idx].y * h
        lo_x, lo_y = lm[left_eye_outer_idx].x * w,  lm[left_eye_outer_idx].y * h
        ri_x, ri_y = lm[right_eye_inner_idx].x * w, lm[right_eye_inner_idx].y * h
        ro_x, ro_y = lm[right_eye_outer_idx].x * w, lm[right_eye_outer_idx].y * h

        pupil_dist = math.hypot(lc_x - rc_x, lc_y - rc_y)
        left_w     = math.hypot(li_x - lo_x, li_y - lo_y)
        right_w    = math.hypot(ri_x - ro_x, ri_y - ro_y)
        ratio      = pupil_dist / (left_w + right_w) if (left_w + right_w) else 0

        cv2.line(img, (int(lc_x), int(lc_y)), (int(rc_x), int(rc_y)), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (int(li_x), int(li_y)), (int(lo_x), int(lo_y)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, (int(ri_x), int(ri_y)), (int(ro_x), int(ro_y)), (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(img, f"Pupil/Width: {ratio:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: 2.00", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def show_mouth_nose_ratio_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        lmx, lmy = lm[left_mouth_corner_idx].x * w, lm[left_mouth_corner_idx].y * h
        rmx, rmy = lm[right_mouth_corner_idx].x * w, lm[right_mouth_corner_idx].y * h
        lnx, lny = lm[left_nose_wing_idx].x * w,    lm[left_nose_wing_idx].y * h
        rnx, rny = lm[right_nose_wing_idx].x * w,   lm[right_nose_wing_idx].y * h

        mouth_w = math.hypot(rmx - lmx, rmy - lmy)
        nose_w  = math.hypot(rnx - lnx, rny - lny)
        ratio   = mouth_w / nose_w if nose_w else 0

        cv2.line(img, (int(lmx), int(lmy)), (int(rmx), int(rmy)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, (int(lnx), int(lny)), (int(rnx), int(rny)), (0, 165, 255), 2, cv2.LINE_AA)

        cv2.putText(img, f"Mouth/Nose: {ratio:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: 1.618", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def show_face_sections_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        flm = results.multi_face_landmarks[0]
        lm = flm.landmark
        top_f = calculate_highest_point_with_percentage_offset(flm, h, w)
        ex = (lm[left_inner_eyebrow_idx].x + lm[right_inner_eyebrow_idx].x) / 2
        ey = (lm[left_inner_eyebrow_idx].y + lm[right_inner_eyebrow_idx].y) / 2
        eb_mid = (int(ex * w), int(ey * h))
        ntip = (int(lm[nose_tip_idx].x * w), int(lm[nose_tip_idx].y * h))
        ctip = (int(lm[chin_idx].x * w),     int(lm[chin_idx].y * h))

        s1 = math.hypot(top_f[0] - eb_mid[0],    top_f[1] - eb_mid[1])
        s2 = math.hypot(eb_mid[0] - ntip[0],     eb_mid[1] - ntip[1])
        s3 = math.hypot(ntip[0] - ctip[0],       ntip[1] - ctip[1])

        cv2.line(img, top_f,    eb_mid,  (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, eb_mid,   ntip,   (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, ntip,     ctip,   (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(img, f"Üst Yüz: {s1:.2f}",    (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Orta Yüz: {s2:.2f}",   (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Alt Yüz: {s3:.2f}",    (20, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: 1:1:1",           (20, 90),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def show_jawline_to_face_width_ratio_on(frame):
    img = frame.copy()
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        jaw_idx = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152,
                   377, 400, 379, 365, 397, 288, 361, 454]
        jaw_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in jaw_idx]

        jaw_len = sum(
            math.hypot(jaw_pts[i+1][0] - jaw_pts[i][0],
                       jaw_pts[i+1][1] - jaw_pts[i][1])
            for i in range(len(jaw_pts)-1)
        )
        left_jaw  = jaw_pts[0]
        right_jaw = jaw_pts[-1]
        face_w = math.hypot(left_jaw[0] - right_jaw[0], left_jaw[1] - right_jaw[1])
        ratio = jaw_len / face_w if face_w else 0

        for i in range(len(jaw_pts)-1):
            cv2.line(img, jaw_pts[i], jaw_pts[i+1], (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, left_jaw, right_jaw, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img, f"Jaw/Face W: {ratio:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Norm: ~2.00", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img

# ─── “calculate_*_on(frame)” routines: return numeric value only (no drawing) ──
def calculate_nose_lips_chin_ratio_on(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 0.0

    lm = results.multi_face_landmarks[0].landmark
    bn_x, bn_y = lm[bottom_nose_idx].x * w, lm[bottom_nose_idx].y * h
    tl_x, tl_y = lm[top_lip_idx].x * w,    lm[top_lip_idx].y * h
    bl_x, bl_y = lm[bottom_lip_idx].x * w, lm[bottom_lip_idx].y * h
    ch_x, ch_y = lm[chin_idx].x * w,       lm[chin_idx].y * h

    d1 = math.hypot(bl_x - ch_x, bl_y - ch_y)
    d2 = math.hypot(bn_x - tl_x, bn_y - tl_y)
    return d1 / d2 if d2 else 0.0

def calculate_eye_symmetry_ratio_on(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 0.0

    lm = results.multi_face_landmarks[0].landmark
    lx, ly = lm[left_eye_inner_idx].x * w,  lm[left_eye_inner_idx].y * h
    rx, ry = lm[right_eye_inner_idx].x * w, lm[right_eye_inner_idx].y * h
    gx, gy = lm[glabella_idx].x * w,       lm[glabella_idx].y * h

    d_l = math.hypot(lx - gx, ly - gy)
    d_r = math.hypot(rx - gx, ry - gy)
    return d_l / d_r if d_r else 0.0

def calculate_face_height_to_width_ratio_on(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 0.0

    flm = results.multi_face_landmarks[0]
    lm = flm.landmark
    highest_x, highest_y = calculate_highest_point_with_percentage_offset(flm, h, w)

    chin = lm[chin_idx]
    ch_x, ch_y = chin.x * w, chin.y * h

    left  = lm[leftmost_face_idx]
    right = lm[rightmost_face_idx]
    l_x, l_y, r_x, r_y = left.x * w, left.y * h, right.x * w, right.y * h

    dh = math.hypot(highest_x - ch_x, highest_y - ch_y)
    dw = math.hypot(l_x - r_x,     l_y - r_y)
    return dh / dw if dw else 0.0

def calculate_cd_ratio_on(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 0.0

    flm = results.multi_face_landmarks[0]
    lm = flm.landmark
    highest_x, highest_y = calculate_highest_point_with_percentage_offset(flm, h, w)

    nose = lm[nose_tip_idx]
    chin = lm[chin_tip_idx]
    nx, ny = nose.x * w, nose.y * h
    cx, cy = chin.x * w, chin.y * h
    C = math.hypot(nx - cx, ny - cy)

    le = lm[left_inner_eyebrow_idx]
    re = lm[right_inner_eyebrow_idx]
    lx, ly = le.x * w, le.y * h
    rx, ry = re.x * w, re.y * h
    mid_x, mid_y = (lx + rx) / 2, (ly + ry) / 2

    D = math.hypot(highest_x - mid_x, highest_y - mid_y)
    return C / D if D else 0.0

def calculate_eye_pupil_to_width_ratio_on(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 0.0

    lm = results.multi_face_landmarks[0].landmark
    lc_x, lc_y = lm[left_eye_center_idx].x * w,  lm[left_eye_center_idx].y * h
    rc_x, rc_y = lm[right_eye_center_idx].x * w, lm[right_eye_center_idx].y * h
    li_x, li_y = lm[left_eye_inner_idx].x * w,  lm[left_eye_inner_idx].y * h
    lo_x, lo_y = lm[left_eye_outer_idx].x * w,  lm[left_eye_outer_idx].y * h
    ri_x, ri_y = lm[right_eye_inner_idx].x * w, lm[right_eye_inner_idx].y * h
    ro_x, ro_y = lm[right_eye_outer_idx].x * w, lm[right_eye_outer_idx].y * h

    pupil_d = math.hypot(lc_x - rc_x, lc_y - rc_y)
    left_w  = math.hypot(li_x - lo_x, li_y - lo_y)
    right_w = math.hypot(ri_x - ro_x, ri_y - ro_y)
    total_w = left_w + right_w
    return pupil_d / total_w if total_w else 0.0

def calculate_mouth_nose_ratio_on(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 0.0

    lm = results.multi_face_landmarks[0].landmark
    lmx, lmy = lm[left_mouth_corner_idx].x * w, lm[left_mouth_corner_idx].y * h
    rmx, rmy = lm[right_mouth_corner_idx].x * w, lm[right_mouth_corner_idx].y * h
    lnx, lny = lm[left_nose_wing_idx].x * w,    lm[left_nose_wing_idx].y * h
    rnx, rny = lm[right_nose_wing_idx].x * w,   lm[right_nose_wing_idx].y * h

    mouth_w = math.hypot(rmx - lmx, rmy - lmy)
    nose_w  = math.hypot(rnx - lnx, rny - lny)
    return mouth_w / nose_w if nose_w else 0.0

def calculate_jawline_to_face_width_ratio_on(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 0.0

    lm = results.multi_face_landmarks[0].landmark
    jaw_idx = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152,
               377, 400, 379, 365, 397, 288, 361, 454]
    jaw_pts = [(lm[i].x * w, lm[i].y * h) for i in jaw_idx]

    jaw_len = sum(
        math.hypot(jaw_pts[i+1][0] - jaw_pts[i][0],
                   jaw_pts[i+1][1] - jaw_pts[i][1])
        for i in range(len(jaw_pts)-1)
    )
    left_jaw  = jaw_pts[0]
    right_jaw = jaw_pts[-1]
    face_w = math.hypot(left_jaw[0] - right_jaw[0], left_jaw[1] - right_jaw[1])
    return jaw_len / face_w if face_w else 0.0

def process_frame(frame):
    """
    Draw exactly one overlay (based on selected_overlay.get())
    onto the given BGR `frame`, then return the annotated frame.
    """
    choice = selected_overlay.get()
    img = frame.copy()

    if choice == "nlc":
        return show_nose_lips_chin_ratio_on(img)
    elif choice == "eye":
        return show_eye_symmetry_ratio_on(img)
    elif choice == "both":
        return show_both_ratios_on(img)
    elif choice == "hwr":
        return show_face_height_to_width_ratio_on(img)
    elif choice == "cd":
        return show_cd_ratio_on(img)
    elif choice == "pupil":
        return show_eye_pupil_to_width_ratio_on(img)
    elif choice == "mouth":
        return show_mouth_nose_ratio_on(img)
    elif choice == "secs":
        return show_face_sections_on(img)
    elif choice == "jaw":
        return show_jawline_to_face_width_ratio_on(img)
    else:
        # "none" or any unexpected value: return unmodified frame
        return img

# ─── display_image: reuse one canvas‐image item so Tkinter won’t lose it ───
_last_tk_image = None
def display_image(image):
    global _last_tk_image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tk_img = ImageTk.PhotoImage(pil_img)
    _last_tk_image = tk_img

    if not hasattr(image_canvas, "image_id"):
        image_canvas.image_id = image_canvas.create_image(
            0, 0, anchor="nw", image=_last_tk_image
        )
    else:
        image_canvas.itemconfig(image_canvas.image_id, image=_last_tk_image)

    image_canvas.config(scrollregion=image_canvas.bbox("all"))

# ─── Global to hold the latest camera frame ─────────────────────────────────◀ NEW
latest_frame = None
# ◀ NEW END

# ─── Camera control ────────────────────────────────────────────────────────
cap = None
def start_camera():
    global cap, latest_frame
    if cap is not None and cap.isOpened():
        return  # already started

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        cap = None
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def update_frame():
        global cap, latest_frame
        if cap is None or not cap.isOpened():
            return

        ret, frame = cap.read()
        if not ret:
            stop_camera()
            return

        frame = cv2.flip(frame, 1)  # optional mirror
        annotated = process_frame(frame)
        latest_frame = frame.copy()   # ◀ NEW: store raw frame for reporting
        display_image(annotated)
        root.after(16, update_frame)

    update_frame()

def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

from PIL import Image, ImageTk
import tkinter.messagebox

from PIL import Image

from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# We'll use Mediapipe's drawing utilities to render the mesh connections:
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh  = mp.solutions.face_mesh

from PIL import Image, ImageTk
import tkinter as tk
import tkinter.messagebox
import cv2
import mediapipe as mp

# Use MediaPipe’s drawing utilities and FaceMesh
mp_drawing   = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Assume `face_mesh` global is already initialized like this somewhere above:
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )

from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

mp_drawing   = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def build_landmark_overlay(frame):
    """
    Takes `frame` (BGR numpy array), runs FaceMesh, and draws all 468 landmarks
    (tesselation) as thin dark‐blue lines on top of the original. Returns a PIL.Image
    at the same resolution (or downscaled if wider than 800px).
    """
    img = frame.copy()           # BGR
    h, w, _ = img.shape

    # 1) Run FaceMesh on an RGB copy
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # 2) If no face is detected, just return the plain frame as PIL:
    if not results.multi_face_landmarks:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 3) Draw the tesselated mesh in dark blue (#003366) on top of the original:
    overlay = img.copy()
    face_landmarks = results.multi_face_landmarks[0]
    dark_blue_bgr = (102, 51, 0)  # BGR for hex #003366

    mp_drawing.draw_landmarks(
        image=overlay,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=dark_blue_bgr,
            thickness=1,
            circle_radius=1
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=dark_blue_bgr,
            thickness=1,
            circle_radius=1
        )
    )

    # 4) Convert BGR→RGB and to PIL.Image
    rgb_out = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_out)

    # 5) If the image is wider than 800px, downscale to max width=800, preserving aspect ratio:
    max_w = 800
    if w > max_w:
        new_h = int(h * (max_w / w))
        pil_img = pil_img.resize((max_w, new_h), Image.LANCZOS)

    return pil_img

    # 5) If the frame is very large, optionally downscale it so it fits on screen.
    #    For example, target a max width of 800px. Adjust as needed.
    max_w = 800
    if w > max_w:
        new_h = int(h * (max_w / w))
        pil_img = pil_img.resize((max_w, new_h), Image.LANCZOS)

    return pil_img



# ─── generate_report() with image ABOVE the text ─────────────────────────────
def generate_report():
    global latest_frame, _last_tk_report_image

    if latest_frame is None:
        tkinter.messagebox.showwarning("Uyarı", "Önce kamerayı başlatın.")
        return

    # 1) Compute all seven ratios on the most recent frame
    r_nlc   = calculate_nose_lips_chin_ratio_on(latest_frame)
    r_eye   = calculate_eye_symmetry_ratio_on(latest_frame)
    r_hwr   = calculate_face_height_to_width_ratio_on(latest_frame)
    r_cd    = calculate_cd_ratio_on(latest_frame)
    r_pupil = calculate_eye_pupil_to_width_ratio_on(latest_frame)
    r_mouth = calculate_mouth_nose_ratio_on(latest_frame)
    r_jaw   = calculate_jawline_to_face_width_ratio_on(latest_frame)

    # 2) Build a list of (name, actual, ideal, info)
    entries = [
        ("Burun-Dudak-Çene Oranı",       r_nlc,   IDEAL_VALUES["Burun-Dudak-Çene Oranı"],       RATIO_INFO["Burun-Dudak-Çene Oranı"]),
        ("Göz Simetrisi Oranı",          r_eye,   IDEAL_VALUES["Göz Simetrisi Oranı"],          RATIO_INFO["Göz Simetrisi Oranı"]),
        ("Yüz H/W Oranı",                r_hwr,   IDEAL_VALUES["Yüz H/W Oranı"],                RATIO_INFO["Yüz H/W Oranı"]),
        ("Burun-Çene/Alın Oranı",        r_cd,    IDEAL_VALUES["Burun-Çene/Alın Oranı"],        RATIO_INFO["Burun-Çene/Alın Oranı"]),
        ("Göz Bebeği/Genislik Oranı",     r_pupil, IDEAL_VALUES["Göz Bebeği/Genislik Oranı"],     RATIO_INFO["Göz Bebeği/Genislik Oranı"]),
        ("Ağız/Burun Oranı",             r_mouth, IDEAL_VALUES["Ağız/Burun Oranı"],             RATIO_INFO["Ağız/Burun Oranı"]),
        ("Çene-Yüz Yükseklik-Genişlik",   r_jaw,   IDEAL_VALUES["Çene-Yüz Yükseklik-Genişlik"],   RATIO_INFO["Çene-Yüz Yükseklik-Genişlik"])
    ]

    # 3) Build the report text with classifications
    report_lines = []
    report_lines.append("Yüz Simetrisi Anlık Raporu")
    report_lines.append("="*40 + "\n")

    deviations = []
    for name, actual, ideal, info in entries:
        dev_percent = abs((actual - ideal) / ideal) * 100 if ideal else 0
        deviations.append(dev_percent)

        if dev_percent < 2:
            status = "Mükemmel"
        elif dev_percent < 5:
            status = "İyi"
        elif dev_percent < 10:
            status = "Normal’e yakın"
        else:
            status = "Yüksek" if actual > ideal else "Düşük"

        report_lines.append(f"{name}:")
        report_lines.append(f"  - Gerçek Değer: {actual:.2f}")
        report_lines.append(f"  - İdeal Değer : {ideal:.3f}")
        report_lines.append(f"  - Sapma       : {dev_percent:.2f}% ({status})")
        report_lines.append(f"  - Bilgi       : {info}\n")

    avg_dev      = sum(deviations) / len(deviations) if deviations else 0
    golden_score = 100 - avg_dev
    report_lines.append("="*40)
    report_lines.append(f"Altın Oran Skoru: {golden_score:.2f}%")
    report_text = "\n".join(report_lines)

    # 4) Build the overlay image (PIL) at its real (or downscaled) size, then convert to Tk.PhotoImage
    pil_overlay = build_landmark_overlay(latest_frame)
    tk_report_img = ImageTk.PhotoImage(pil_overlay)
    _last_tk_report_image = tk_report_img  # keep a reference so Tkinter doesn’t trash it

    # 5) Create a vertical layout: image on top, text below
    popup = tk.Toplevel(root)
    popup.title("Anlık Analiz Raporu")

    # Image Frame (top)
    img_frame = tk.Frame(popup)
    img_frame.pack(side="top", fill="both", expand=False, padx=5, pady=5)
    img_label = tk.Label(img_frame, image=tk_report_img)
    img_label.pack()

    # Text Frame (below), with vertical scrolling
    text_frame = tk.Frame(popup)
    text_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)

    text_widget = tk.Text(text_frame, wrap="word", width=60, height=20, padx=10, pady=10)
    text_widget.insert("1.0", report_text)
    text_widget.config(state="disabled")
    text_widget.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    text_widget.configure(yscrollcommand=scrollbar.set)

    # Optionally, you can set a minimum size for the popup so that the
    # full‐res image is visible without immediate cropping:
    popup.minsize(pil_overlay.width + 20, pil_overlay.height + 200)

    # … your existing code that creates popup, img_label and text_widget …

    # ─── Download Button ───────────────────────────────
    def on_download():
        # popup variables are in this scope:
        export_pdf(pil_overlay, report_text,
                   f"YuzRapor_{time.strftime('%Y%m%d_%H%M%S')}.pdf")

    download_btn = tk.Button(popup, text="Download as PDF", command=on_download)
    download_btn.pack(pady=5)


# ─── Build the Tkinter UI ───────────────────────────────────────────────────
root = tk.Tk()
root.title("Facial Symmetry Ratio Calculator")

# --- SINGLE StringVar that holds the currently selected overlay ---
selected_overlay = tk.StringVar(value="none")

# === Radiobuttons: Only one can be active at a time ===
rb_nlc = tk.Radiobutton(
    root,
    text="Nose-Lips-Chin Ratio",
    variable=selected_overlay,
    value="nlc"
)
rb_nlc.pack(anchor="w")

rb_eye = tk.Radiobutton(
    root,
    text="Eye Symmetry Ratio",
    variable=selected_overlay,
    value="eye"
)
rb_eye.pack(anchor="w")

rb_both = tk.Radiobutton(
    root,
    text="Both Ratios",
    variable=selected_overlay,
    value="both"
)
rb_both.pack(anchor="w")

rb_hwr = tk.Radiobutton(
    root,
    text="Face Height/Width Ratio",
    variable=selected_overlay,
    value="hwr"
)
rb_hwr.pack(anchor="w")

rb_cd = tk.Radiobutton(
    root,
    text="Nose-Chin/Forehead (C/D) Ratio",
    variable=selected_overlay,
    value="cd"
)
rb_cd.pack(anchor="w")

rb_pupil = tk.Radiobutton(
    root,
    text="Eye Pupil/Width Ratio",
    variable=selected_overlay,
    value="pupil"
)
rb_pupil.pack(anchor="w")

rb_mouth = tk.Radiobutton(
    root,
    text="Mouth/Nose Ratio",
    variable=selected_overlay,
    value="mouth"
)
rb_mouth.pack(anchor="w")

rb_secs = tk.Radiobutton(
    root,
    text="Face Sections",
    variable=selected_overlay,
    value="secs"
)
rb_secs.pack(anchor="w")

rb_jaw = tk.Radiobutton(
    root,
    text="Jawline/Face Width Ratio",
    variable=selected_overlay,
    value="jaw"
)
rb_jaw.pack(anchor="w")

# ─── Canvas + Scrollbars ──────────────────────────────────────────────────
canvas_frame = tk.Frame(root)
canvas_frame.pack(fill="both", expand=True)

image_canvas = tk.Canvas(canvas_frame, width=600, height=600)
image_canvas.pack(side="left", fill="both", expand=True)

scroll_x = tk.Scrollbar(canvas_frame, orient="horizontal", command=image_canvas.xview)
scroll_x.pack(side="bottom", fill="x")
scroll_y = tk.Scrollbar(canvas_frame, orient="vertical",   command=image_canvas.yview)
scroll_y.pack(side="right", fill="y")

image_canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)

# ─── Camera Buttons ─────────────────────────────────────────────────────────
tk.Button(root, text="Start Camera",    command=start_camera).pack(pady=5)
tk.Button(root, text="Stop Camera",     command=stop_camera).pack(pady=5)

# ─── Generate Report Button ◀ NEW ──────────────────────────────────────────
tk.Button(root, text="Generate Report", command=generate_report).pack(pady=5)
# ◀ NEW END

root.mainloop()
