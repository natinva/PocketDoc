import cv2
import mediapipe as mp
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import datetime

# -------------------- GLOBALS --------------------

loaded_image = None
width = 0
height = 0
_cached_landmarks = None
report_text_widget = None  # bottom-right analysis box

# -------------------- FONT REGISTRATION --------------------

SCRIPT_DIR = os.path.dirname(__file__)
FONT_PATH = os.path.abspath(os.path.join(
    SCRIPT_DIR,
    os.pardir,
    os.pardir,
    "Fonts", "League_Spartan", "static",
    "LeagueSpartan-SemiBold.ttf"
))

try:
    pdfmetrics.registerFont(
        TTFont("LeagueSpartan-SemiBold", FONT_PATH)
    )
    PDF_FONT_NAME = "LeagueSpartan-SemiBold"
except Exception:
    PDF_FONT_NAME = "Helvetica"

# -------------------- MEDIAPIPE SETUP --------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

# Landmark indices
bottom_nose_idx = 2
top_lip_idx = 13
bottom_lip_idx = 14
chin_idx = 152
left_eye_inner_idx = 133
right_eye_inner_idx = 362
glabella_idx = 168
top_forehead_idx = 10
leftmost_face_idx = 234
rightmost_face_idx = 454
nose_tip_idx = 1
chin_tip_idx = 152
left_inner_eyebrow_idx = 55
right_inner_eyebrow_idx = 285
left_eye_center_idx = 468
right_eye_center_idx = 473
left_eye_outer_idx = 33
right_eye_outer_idx = 263
left_mouth_corner_idx = 61
right_mouth_corner_idx = 291
left_nose_wing_idx = 49
right_nose_wing_idx = 279

# -------------------- BASIC HELPERS --------------------

def euclidean_2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def euclidean_px(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_highest_point_with_percentage_offset(face_landmarks, img_height, img_width, offset_ratio=0.1):
    top_forehead = face_landmarks.landmark[top_forehead_idx]
    chin = face_landmarks.landmark[chin_idx]

    top_forehead_x = int(top_forehead.x * img_width)
    top_forehead_y = int(top_forehead.y * img_height)
    chin_y = int(chin.y * img_height)

    face_height = chin_y - top_forehead_y
    offset_in_pixels = int(face_height * offset_ratio)

    highest_y = max(0, top_forehead_y - offset_in_pixels)
    return top_forehead_x, highest_y


def get_face_landmarks():
    global _cached_landmarks, loaded_image

    if loaded_image is None:
        return None

    if _cached_landmarks is not None:
        return _cached_landmarks

    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        _cached_landmarks = results.multi_face_landmarks[0]
        return _cached_landmarks

    return None

# -------------------- LOAD & DISPLAY --------------------

def load_image(image_path):
    global loaded_image, width, height, _cached_landmarks
    loaded_image = cv2.imread(image_path)
    if loaded_image is None:
        messagebox.showerror("Error", "Image could not be loaded.")
        return
    height, width, _ = loaded_image.shape
    _cached_landmarks = None


def display_image(image):
    """
    Show the given BGR image in the top-right canvas.
    Uses 'contain' logic: whole image visible and centered.
    """
    if image is None:
        return

    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(processed_image)

    # Current canvas size
    image_canvas.update_idletasks()
    canvas_w = image_canvas.winfo_width()
    canvas_h = image_canvas.winfo_height()
    if canvas_w <= 1 or canvas_h <= 1:
        return

    img_w, img_h = pil_image.size
    # scale so that the whole image fits inside the canvas
    scale = min(canvas_w / img_w, canvas_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2

    tk_image = ImageTk.PhotoImage(pil_image)
    image_canvas.delete("all")
    image_canvas.create_image(offset_x, offset_y, anchor="nw", image=tk_image)
    image_canvas.image = tk_image


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        load_image(file_path)
        if loaded_image is not None:
            display_image(loaded_image)
            update_report_text(
                "Image loaded.\nClick any analysis button or 'Download PDF Report' "
                "to generate a full golden ratio report."
            )

# -------------------- RATIO CALCULATIONS --------------------

def calc_nose_lips_chin_ratio(landmarks):
    bottom_nose = landmarks.landmark[bottom_nose_idx]
    top_lip = landmarks.landmark[top_lip_idx]
    bottom_lip = landmarks.landmark[bottom_lip_idx]
    chin = landmarks.landmark[chin_idx]

    bottom_lip_to_chin = euclidean_2d(bottom_lip, chin)
    nose_to_top_lip = euclidean_2d(bottom_nose, top_lip)

    return bottom_lip_to_chin / nose_to_top_lip if nose_to_top_lip != 0 else 0.0


def calc_eye_symmetry_ratio(landmarks):
    left_eye_inner = landmarks.landmark[left_eye_inner_idx]
    right_eye_inner = landmarks.landmark[right_eye_inner_idx]
    glabella = landmarks.landmark[glabella_idx]

    left_dist = euclidean_2d(left_eye_inner, glabella)
    right_dist = euclidean_2d(right_eye_inner, glabella)

    return left_dist / right_dist if right_dist != 0 else 0.0


def calc_face_height_to_width_ratio(landmarks):
    highest_x, highest_y = calculate_highest_point_with_percentage_offset(landmarks, height, width)

    chin = landmarks.landmark[chin_idx]
    leftmost_face = landmarks.landmark[leftmost_face_idx]
    rightmost_face = landmarks.landmark[rightmost_face_idx]

    chin_px = (chin.x * width, chin.y * height)
    left_px = (leftmost_face.x * width, leftmost_face.y * height)
    right_px = (rightmost_face.x * width, rightmost_face.y * height)

    face_height = euclidean_px((highest_x, highest_y), chin_px)
    face_width = euclidean_px(left_px, right_px)

    return face_height / face_width if face_width != 0 else 0.0


def calc_cd_ratio(landmarks):
    highest_x, highest_y = calculate_highest_point_with_percentage_offset(landmarks, height, width)

    nose_tip = landmarks.landmark[nose_tip_idx]
    chin_tip = landmarks.landmark[chin_tip_idx]
    left_eyebrow = landmarks.landmark[left_inner_eyebrow_idx]
    right_eyebrow = landmarks.landmark[right_inner_eyebrow_idx]

    nose_px = (nose_tip.x * width, nose_tip.y * height)
    chin_px = (chin_tip.x * width, chin_tip.y * height)
    midpoint_x = (left_eyebrow.x + right_eyebrow.x) / 2 * width
    midpoint_y = (left_eyebrow.y + right_eyebrow.y) / 2 * height
    eyebrow_mid_px = (midpoint_x, midpoint_y)

    C = euclidean_px(nose_px, chin_px)
    D = euclidean_px((highest_x, highest_y), eyebrow_mid_px)

    return C / D if D != 0 else 0.0


def calc_eye_distance_width_ratio(landmarks):
    left_eye_center = landmarks.landmark[left_eye_center_idx]
    right_eye_center = landmarks.landmark[right_eye_center_idx]
    left_eye_inner = landmarks.landmark[left_eye_inner_idx]
    left_eye_outer = landmarks.landmark[left_eye_outer_idx]
    right_eye_inner = landmarks.landmark[right_eye_inner_idx]
    right_eye_outer = landmarks.landmark[right_eye_outer_idx]

    eye_distance = euclidean_2d(left_eye_center, right_eye_center)
    left_eye_width = euclidean_2d(left_eye_inner, left_eye_outer)
    right_eye_width = euclidean_2d(right_eye_inner, right_eye_outer)
    mean_width = (left_eye_width + right_eye_width) / 2.0

    return eye_distance / mean_width if mean_width != 0 else 0.0


def calc_mouth_nose_ratio(landmarks):
    left_mouth_corner = landmarks.landmark[left_mouth_corner_idx]
    right_mouth_corner = landmarks.landmark[right_mouth_corner_idx]
    left_nose_wing = landmarks.landmark[left_nose_wing_idx]
    right_nose_wing = landmarks.landmark[right_nose_wing_idx]

    mouth_width = euclidean_2d(left_mouth_corner, right_mouth_corner)
    nose_width = euclidean_2d(left_nose_wing, right_nose_wing)

    return mouth_width / nose_width if nose_width != 0 else 0.0


def calc_jawline_to_face_width_ratio(landmarks):
    jawline_indices = [234, 93, 132, 58, 172, 136, 150, 176, 148,
                       152, 377, 400, 379, 365, 397, 288, 361, 454]
    jaw_points_px = [
        (landmarks.landmark[idx].x * width,
         landmarks.landmark[idx].y * height)
        for idx in jawline_indices
    ]

    jawline_length = 0.0
    for i in range(len(jaw_points_px) - 1):
        jawline_length += euclidean_px(jaw_points_px[i], jaw_points_px[i + 1])

    face_width = euclidean_px(jaw_points_px[0], jaw_points_px[-1])

    return jawline_length / face_width if face_width != 0 else 0.0

# -------------------- IDEAL VALUES --------------------

IDEAL_VALUES = {
    "Nose–Lips–Chin Ratio": 1.618,
    "Eye Symmetry Ratio": 1.0,
    "Face Height-to-Width Ratio": 1.618,
    "Nose–Chin / Forehead Ratio": 1.618,
    "Eye Distance / Mean Eye Width Ratio": 2.0,
    "Mouth / Nose Ratio": 1.618,
    "Jawline-to-Face Width Ratio": 2.0
}

# -------------------- PDF EXPORT --------------------

def export_pdf(image: Image.Image, report: str, filename="FaceAnalysisReport.pdf"):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    img_reader = ImageReader(buf)
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont(PDF_FONT_NAME, 12)

    pw, ph = letter
    iw, ih = image.size

    max_w, max_h = pw / 2 - 40, ph - 80
    scale = min(max_w / iw, max_h / ih)
    dw, dh = iw * scale, ih * scale

    c.drawImage(img_reader, 20, ph - dh - 40, width=dw, height=dh)

    text = c.beginText(pw / 2 + 20, ph - 40)
    text.setLeading(14)
    for line in report.splitlines():
        text.textLine(line)
    c.drawText(text)

    c.save()
    messagebox.showinfo("PDF Saved", f"Report saved as:\n{filename}")

# -------------------- ANALYSIS + REPORT --------------------

def analyze_and_prepare_report():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return None, None

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected. Please try a clearer frontal face photo.")
        return None, None

    ratios = {
        "Nose–Lips–Chin Ratio": calc_nose_lips_chin_ratio(landmarks),
        "Eye Symmetry Ratio": calc_eye_symmetry_ratio(landmarks),
        "Face Height-to-Width Ratio": calc_face_height_to_width_ratio(landmarks),
        "Nose–Chin / Forehead Ratio": calc_cd_ratio(landmarks),
        "Eye Distance / Mean Eye Width Ratio": calc_eye_distance_width_ratio(landmarks),
        "Mouth / Nose Ratio": calc_mouth_nose_ratio(landmarks),
        "Jawline-to-Face Width Ratio": calc_jawline_to_face_width_ratio(landmarks)
    }

    deviations = []
    report = "Facial Symmetry & Golden Ratio Analysis Report\n" + "=" * 50 + "\n\n"

    for name, actual in ratios.items():
        ideal = IDEAL_VALUES.get(name, 0.0)
        dev = abs((actual - ideal) / ideal) * 100 if ideal != 0 else 0.0
        deviations.append(dev)

        report += (
            f"{name}:\n"
            f"  • Actual Value: {actual:.2f}\n"
            f"  • Ideal Value:  {ideal:.3f}\n"
            f"  • Deviation:    {dev:.2f}%\n\n"
        )

    avg_dev = sum(deviations) / len(deviations) if deviations else 0.0
    score = max(0.0, 100.0 - avg_dev)

    report += "=" * 50 + "\n"
    report += f"Overall Golden Ratio Score: {score:.2f}%\n"
    report += "=" * 50 + "\n\n"
    report += (
        "Note: This analysis is based on geometric facial\n"
        "proportions and does not replace\n"
        "professional aesthetic consultation or clinical evaluation.\n"
    )

    img_land = loaded_image.copy()
    h_img, w_img = img_land.shape[:2]
    for lm in landmarks.landmark:
        x, y = int(lm.x * w_img), int(lm.y * h_img)
        cv2.circle(img_land, (x, y), 1, (255, 255, 255), -1)

    pil_img = Image.fromarray(cv2.cvtColor(img_land, cv2.COLOR_BGR2RGB))

    return report, pil_img


def update_report_text(text):
    if report_text_widget is None:
        return
    report_text_widget.config(state="normal")
    report_text_widget.delete("1.0", tk.END)
    report_text_widget.insert(tk.END, text)
    report_text_widget.config(state="disabled")


def on_download_report():
    report, pil_img = analyze_and_prepare_report()
    if report is None:
        return

    update_report_text(report)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FaceAnalysis_{timestamp}.pdf"
    export_pdf(pil_img, report, filename)

# -------------------- VISUALIZATION BUTTONS --------------------

def show_nose_lips_chin_ratio():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    bottom_nose = landmarks.landmark[bottom_nose_idx]
    top_lip = landmarks.landmark[top_lip_idx]
    bottom_lip = landmarks.landmark[bottom_lip_idx]
    chin = landmarks.landmark[chin_idx]

    bottom_nose_px = (int(bottom_nose.x * width), int(bottom_nose.y * height))
    top_lip_px = (int(top_lip.x * width), int(top_lip.y * height))
    bottom_lip_px = (int(bottom_lip.x * width), int(bottom_lip.y * height))
    chin_px = (int(chin.x * width), int(chin.y * height))

    ratio = calc_nose_lips_chin_ratio(landmarks)

    cv2.line(image, bottom_nose_px, top_lip_px, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.line(image, bottom_lip_px, chin_px, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.putText(image, f"Nose–Lips–Chin: {ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: 1.618", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    display_image(image)


def show_eye_symmetry_ratio():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    left_eye_inner = landmarks.landmark[left_eye_inner_idx]
    right_eye_inner = landmarks.landmark[right_eye_inner_idx]
    glabella = landmarks.landmark[glabella_idx]

    left_eye_px = (int(left_eye_inner.x * width), int(left_eye_inner.y * height))
    right_eye_px = (int(right_eye_inner.x * width), int(right_eye_inner.y * height))
    glabella_px = (int(glabella.x * width), int(glabella.y * height))

    ratio = calc_eye_symmetry_ratio(landmarks)

    cv2.line(image, left_eye_px, glabella_px, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, right_eye_px, glabella_px, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.putText(image, f"Eye Symmetry: {ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: 1.000", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    display_image(image)


def show_both_ratios():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    bottom_nose = landmarks.landmark[bottom_nose_idx]
    top_lip = landmarks.landmark[top_lip_idx]
    bottom_lip = landmarks.landmark[bottom_lip_idx]
    chin = landmarks.landmark[chin_idx]

    bottom_nose_px = (int(bottom_nose.x * width), int(bottom_nose.y * height))
    top_lip_px = (int(top_lip.x * width), int(top_lip.y * height))
    bottom_lip_px = (int(bottom_lip.x * width), int(bottom_lip.y * height))
    chin_px = (int(chin.x * width), int(chin.y * height))

    nose_ratio = calc_nose_lips_chin_ratio(landmarks)

    cv2.line(image, bottom_nose_px, top_lip_px, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.line(image, bottom_lip_px, chin_px, (255, 0, 0), 3, cv2.LINE_AA)

    left_eye_inner = landmarks.landmark[left_eye_inner_idx]
    right_eye_inner = landmarks.landmark[right_eye_inner_idx]
    glabella = landmarks.landmark[glabella_idx]

    left_eye_px = (int(left_eye_inner.x * width), int(left_eye_inner.y * height))
    right_eye_px = (int(right_eye_inner.x * width), int(right_eye_inner.y * height))
    glabella_px = (int(glabella.x * width), int(glabella.y * height))

    eye_ratio = calc_eye_symmetry_ratio(landmarks)

    cv2.line(image, left_eye_px, glabella_px, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, right_eye_px, glabella_px, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.putText(image, f"Nose–Lips–Chin: {nose_ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: 1.618", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(image, f"Eye Symmetry: {eye_ratio:.2f}", (20, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: 1.000", (20, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    display_image(image)


def show_face_height_to_width_ratio():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    highest_x, highest_y = calculate_highest_point_with_percentage_offset(landmarks, height, width)
    chin = landmarks.landmark[chin_idx]
    chin_px = (int(chin.x * width), int(chin.y * height))

    leftmost_face = landmarks.landmark[leftmost_face_idx]
    rightmost_face = landmarks.landmark[rightmost_face_idx]
    left_px = (int(leftmost_face.x * width), int(leftmost_face.y * height))
    right_px = (int(rightmost_face.x * width), int(rightmost_face.y * height))

    ratio = calc_face_height_to_width_ratio(landmarks)

    cv2.line(image, (highest_x, highest_y), chin_px, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(image, left_px, right_px, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(image, f"Height / Width: {ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: 1.618", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    display_image(image)


def show_cd_ratio():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    highest_x, highest_y = calculate_highest_point_with_percentage_offset(landmarks, height, width)

    nose_tip = landmarks.landmark[nose_tip_idx]
    chin_tip = landmarks.landmark[chin_tip_idx]
    nose_px = (int(nose_tip.x * width), int(nose_tip.y * height))
    chin_px = (int(chin_tip.x * width), int(chin_tip.y * height))

    left_eyebrow = landmarks.landmark[left_inner_eyebrow_idx]
    right_eyebrow = landmarks.landmark[right_inner_eyebrow_idx]
    midpoint_x = int((left_eyebrow.x + right_eyebrow.x) / 2 * width)
    midpoint_y = int((left_eyebrow.y + right_eyebrow.y) / 2 * height)
    eyebrow_mid_px = (midpoint_x, midpoint_y)

    ratio = calc_cd_ratio(landmarks)

    cv2.line(image, nose_px, chin_px, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.line(image, (highest_x, highest_y), eyebrow_mid_px, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.putText(image, f"Nose–Chin / Forehead: {ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: 1.618", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    display_image(image)


def show_eye_distance_width_ratio():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    left_eye_center = landmarks.landmark[left_eye_center_idx]
    right_eye_center = landmarks.landmark[right_eye_center_idx]
    left_eye_inner = landmarks.landmark[left_eye_inner_idx]
    left_eye_outer = landmarks.landmark[left_eye_outer_idx]
    right_eye_inner = landmarks.landmark[right_eye_inner_idx]
    right_eye_outer = landmarks.landmark[right_eye_outer_idx]

    left_center_px = (int(left_eye_center.x * width), int(left_eye_center.y * height))
    right_center_px = (int(right_eye_center.x * width), int(right_eye_center.y * height))
    left_inner_px = (int(left_eye_inner.x * width), int(left_eye_inner.y * height))
    left_outer_px = (int(left_eye_outer.x * width), int(left_eye_outer.y * height))
    right_inner_px = (int(right_eye_inner.x * width), int(right_eye_inner.y * height))
    right_outer_px = (int(right_eye_outer.x * width), int(right_eye_outer.y * height))

    ratio = calc_eye_distance_width_ratio(landmarks)

    cv2.line(image, left_center_px, right_center_px, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.line(image, left_inner_px, left_outer_px, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, right_inner_px, right_outer_px, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.putText(image, f"Eye Dist / Mean Width: {ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: ~2.0", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    display_image(image)


def show_mouth_nose_ratio():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    left_mouth_corner = landmarks.landmark[left_mouth_corner_idx]
    right_mouth_corner = landmarks.landmark[right_mouth_corner_idx]
    left_nose_wing = landmarks.landmark[left_nose_wing_idx]
    right_nose_wing = landmarks.landmark[right_nose_wing_idx]

    left_mouth_px = (int(left_mouth_corner.x * width), int(left_mouth_corner.y * height))
    right_mouth_px = (int(right_mouth_corner.x * width), int(right_mouth_corner.y * height))
    left_nose_px = (int(left_nose_wing.x * width), int(left_nose_wing.y * height))
    right_nose_px = (int(right_nose_wing.x * width), int(right_nose_wing.y * height))

    ratio = calc_mouth_nose_ratio(landmarks)

    cv2.line(image, left_mouth_px, right_mouth_px, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, left_nose_px, right_nose_px, (0, 165, 255), 3, cv2.LINE_AA)

    cv2.putText(image, f"Mouth / Nose: {ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: 1.618", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    display_image(image)


def show_face_sections():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    highest_x, highest_y = calculate_highest_point_with_percentage_offset(landmarks, height, width)
    left_eyebrow = landmarks.landmark[left_inner_eyebrow_idx]
    right_eyebrow = landmarks.landmark[right_inner_eyebrow_idx]
    eyebrow_mid_x = int((left_eyebrow.x + right_eyebrow.x) / 2 * width)
    eyebrow_mid_y = int((left_eyebrow.y + right_eyebrow.y) / 2 * height)

    nose_tip = landmarks.landmark[nose_tip_idx]
    chin_tip = landmarks.landmark[chin_tip_idx]

    nose_tip_px = (int(nose_tip.x * width), int(nose_tip.y * height))
    chin_tip_px = (int(chin_tip.x * width), int(chin_tip.y * height))

    top_forehead_px = (highest_x, highest_y)
    eyebrow_mid_px = (eyebrow_mid_x, eyebrow_mid_y)

    upper_len = euclidean_px(top_forehead_px, eyebrow_mid_px)
    middle_len = euclidean_px(eyebrow_mid_px, nose_tip_px)
    lower_len = euclidean_px(nose_tip_px, chin_tip_px)

    cv2.line(image, top_forehead_px, eyebrow_mid_px, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.line(image, eyebrow_mid_px, nose_tip_px, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, nose_tip_px, chin_tip_px, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.putText(image, f"Upper Face:  {upper_len:.1f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Middle Face: {middle_len:.1f}", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Lower Face:  {lower_len:.1f}", (20, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal proportion: 1 : 1 : 1", (20, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    display_image(image)


def show_jawline_to_face_width_ratio():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    landmarks = get_face_landmarks()
    if landmarks is None:
        messagebox.showwarning("Warning", "No face detected.")
        return

    image = loaded_image.copy()

    jawline_indices = [234, 93, 132, 58, 172, 136, 150, 176, 148,
                       152, 377, 400, 379, 365, 397, 288, 361, 454]
    jaw_points_px = [
        (int(landmarks.landmark[idx].x * width),
         int(landmarks.landmark[idx].y * height))
        for idx in jawline_indices
    ]

    for i in range(len(jaw_points_px) - 1):
        cv2.line(image, jaw_points_px[i], jaw_points_px[i + 1], (0, 255, 255), 2, cv2.LINE_AA)

    cv2.line(image, jaw_points_px[0], jaw_points_px[-1], (255, 0, 0), 2, cv2.LINE_AA)

    ratio = calc_jawline_to_face_width_ratio(landmarks)

    cv2.putText(image, f"Jawline / Face Width: {ratio:.2f}", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Ideal: ~2.0", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    display_image(image)

# -------------------- TKINTER UI --------------------

root = tk.Tk()
root.title("Facial Symmetry & Golden Ratio Analyzer")

main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_panel = tk.Frame(main_frame)
left_panel.pack(side="left", fill="y", padx=10, pady=10)

right_panel = tk.Frame(main_frame, bg="black")
right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

image_canvas = tk.Canvas(right_panel, bg="black", highlightthickness=0)
image_canvas.pack(side="top", fill="both", expand=True)

report_frame = tk.Frame(right_panel)
report_frame.pack(side="bottom", fill="x")

report_label = tk.Label(report_frame, text="Golden Ratio Analysis", font=("Helvetica", 11, "bold"))
report_label.pack(anchor="w")

report_text_widget = tk.Text(report_frame, height=10, wrap="word", state="disabled")
report_text_widget.pack(fill="x", expand=False)

update_report_text(
    "Load a frontal face image to start.\n"
    "The golden ratio analysis report will appear here."
)

btn_open = tk.Button(left_panel, text="Open Image", width=25, command=open_image)
btn_open.pack(pady=3)

btn_nose_lips_chin = tk.Button(left_panel, text="Show Nose–Lips–Chin Ratio",
                               width=25, command=show_nose_lips_chin_ratio)
btn_nose_lips_chin.pack(pady=3)

btn_eye_symmetry = tk.Button(left_panel, text="Show Eye Symmetry Ratio",
                             width=25, command=show_eye_symmetry_ratio)
btn_eye_symmetry.pack(pady=3)

btn_both = tk.Button(left_panel, text="Show Both (Nose & Eyes)",
                     width=25, command=show_both_ratios)
btn_both.pack(pady=3)

btn_face_hw = tk.Button(left_panel, text="Show Height / Width Ratio",
                        width=25, command=show_face_height_to_width_ratio)
btn_face_hw.pack(pady=3)

btn_cd = tk.Button(left_panel, text="Show Nose–Chin / Forehead",
                   width=25, command=show_cd_ratio)
btn_cd.pack(pady=3)

btn_eye_dist = tk.Button(left_panel, text="Show Eye Distance / Width",
                         width=25, command=show_eye_distance_width_ratio)
btn_eye_dist.pack(pady=3)

btn_mouth_nose = tk.Button(left_panel, text="Show Mouth / Nose Ratio",
                           width=25, command=show_mouth_nose_ratio)
btn_mouth_nose.pack(pady=3)

btn_sections = tk.Button(left_panel, text="Show Face Sections (1:1:1)",
                         width=25, command=show_face_sections)
btn_sections.pack(pady=3)

btn_jawline = tk.Button(left_panel, text="Show Jawline / Face Width",
                        width=25, command=show_jawline_to_face_width_ratio)
btn_jawline.pack(pady=3)

btn_report = tk.Button(left_panel, text="Download PDF Report",
                       width=25, command=on_download_report)
btn_report.pack(pady=10)

root.mainloop()
