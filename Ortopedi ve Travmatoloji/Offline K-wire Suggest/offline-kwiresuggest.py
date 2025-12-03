import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
import textwrap
import numpy as np

# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "/Users/avnitan/PycharmProjects/TestProject/PocketDoc/Modeller/Orthopaedics and Traumatology/Supracondylar Humerus - AP Xray/k-teli.pt"

BRAND_DARK = "#003366"
BRAND_LIGHT = "#e6f2ff"
BRAND_WHITE = "#ffffff"

# Colors for segmentation / drawing (BGR)
COL_HUMERUS = (200, 200, 200)  # light grey
COL_FOSSA   = (255, 0, 0)      # blue
COL_EPIC    = (0, 255, 255)    # yellow
COL_WIRE    = (0, 0, 255)      # red wires


# =========================
# GEOMETRY UTILITIES
# =========================

def compute_angle_between(v1, v2):
    """Return angle (degrees) between 2D vectors v1 and v2."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def extend_line_to_bounds(x0, y0, angle_rad, img_w, img_h, length_factor=3.0):
    """
    Extend a ray from (x0, y0) at angle angle_rad (radians) across the image.
    Returns endpoint clipped to image bounds.
    """
    L = length_factor * math.hypot(img_w, img_h)
    dx = math.cos(angle_rad) * L
    dy = math.sin(angle_rad) * L

    x1 = x0 + dx
    y1 = y0 + dy

    points = []
    if dx != 0:
        t = (0 - x0) / dx
        if 0 <= t <= 1:
            y = y0 + t * dy
            if 0 <= y <= img_h - 1:
                points.append((0, y))
        t = ((img_w - 1) - x0) / dx
        if 0 <= t <= 1:
            y = y0 + t * dy
            if 0 <= y <= img_h - 1:
                points.append((img_w - 1, y))
    if dy != 0:
        t = (0 - y0) / dy
        if 0 <= t <= 1:
            x = x0 + t * dx
            if 0 <= x <= img_w - 1:
                points.append((x, 0))
        t = ((img_h - 1) - y0) / dy
        if 0 <= t <= 1:
            x = x0 + t * dx
            if 0 <= x <= img_h - 1:
                points.append((x, img_h - 1))

    if not points:
        return int(round(x1)), int(round(y1))
    far = max(points, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)
    return int(round(far[0])), int(round(far[1]))


def _box_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# =========================
# MODEL WRAPPER – SEGMENTATION
# =========================

class LandmarkDetector:
    """
    Uses Ultralytics segmentation model.
    - Uses masks for visualization.
    - Derives bounding rectangles from masks for geometry.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None

    def load_model(self):
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image_path):
        model = self.load_model()
        result = model(image_path)[0]

        boxes = result.boxes
        masks = result.masks
        if masks is None:
            raise RuntimeError("Model did not return masks. Make sure this is a segmentation model.")

        cls = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        mask_data = masks.data.cpu().numpy()
        orig_h, orig_w = result.orig_shape

        epic_boxes, epic_masks = [], []
        fossa_boxes, fossa_masks = [], []
        humerus_boxes, humerus_masks = [], []

        for i, (c, box) in enumerate(zip(cls, xyxy)):
            mask_i = mask_data[i]
            mask_i = (mask_i > 0.5).astype(np.uint8)
            if mask_i.shape != (orig_h, orig_w):
                mask_i = cv2.resize(mask_i, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            x, y, w, h = cv2.boundingRect(mask_i)
            bbox = (float(x), float(y), float(x + w), float(y + h))

            if c == 0:   # epicondyle
                epic_boxes.append(bbox)
                epic_masks.append(mask_i)
            elif c == 1: # fossa
                fossa_boxes.append(bbox)
                fossa_masks.append(mask_i)
            elif c == 2: # humerus
                humerus_boxes.append(bbox)
                humerus_masks.append(mask_i)

        landmarks = {
            "epicondyles": epic_boxes,
            "fossa": fossa_boxes[0] if fossa_boxes else None,
            "humerus": humerus_boxes[0] if humerus_boxes else None,
            "masks": {
                "epicondyles": epic_masks,
                "fossa": fossa_masks[0] if fossa_masks else None,
                "humerus": humerus_masks[0] if humerus_masks else None,
            }
        }
        return landmarks


# =========================
# K-WIRE PLANNING
# =========================

def plan_cross_formation(img, landmarks):
    """
    Cross formation (always 2 wires).
    Wires extend across the image (they cross and continue).
    """
    epicondyles = landmarks["epicondyles"]
    fossa = landmarks["fossa"]

    if fossa is None or len(epicondyles) < 2:
        raise ValueError("Cross formation requires at least 2 epicondyles and a fossa.")

    img_h, img_w = img.shape[:2]

    epic_sorted = sorted(epicondyles, key=lambda b: _box_center(b)[0])
    med_box = epic_sorted[0]
    lat_box = epic_sorted[-1]

    med_c = _box_center(med_box)
    lat_c = _box_center(lat_box)

    fx1, fy1, fx2, fy2 = fossa
    f_cx = (fx1 + fx2) / 2.0
    f_h = fy2 - fy1

    apex_y = fy1 - 0.3 * f_h
    if apex_y < 0:
        apex_y = max(5, fy1 * 0.3)
    apex = [f_cx, apex_y]

    angle_deg = None
    for _ in range(40):
        v1 = (med_c[0] - apex[0], med_c[1] - apex[1])
        v2 = (lat_c[0] - apex[0], lat_c[1] - apex[1])
        angle_deg = compute_angle_between(v1, v2)
        if angle_deg is None:
            break
        if angle_deg >= 75 or apex[1] <= 10:
            break
        apex[1] -= 5

    apex = (apex[0], apex[1])

    wires = []
    for ep_c in (med_c, lat_c):
        angle = math.atan2(apex[1] - ep_c[1], apex[0] - ep_c[0])
        end_x, end_y = extend_line_to_bounds(ep_c[0], ep_c[1], angle, img_w, img_h)
        wires.append(
            ((int(round(ep_c[0])), int(round(ep_c[1]))),
             (int(round(end_x)), int(round(end_y))))
        )

    info = {
        "type": "Cross (2 wires)",
        "angle_between_wires_deg": angle_deg,
        "medial_epicondyle": med_c,
        "lateral_epicondyle": lat_c,
        "apex": tuple(apex),
        "num_wires": 2,
    }
    return wires, info


def plan_lateral_formation(img, landmarks):
    """
    Lateral-only formation: ALWAYS 3 wires
    - Entry from lateral epicondyle
    - 3 wires with ~20° spacing around a central wire above fossa.
    """
    num_wires = 3
    epicondyles = landmarks["epicondyles"]
    fossa = landmarks["fossa"]

    if fossa is None or len(epicondyles) < 1:
        raise ValueError("Lateral-only formation requires ≥1 epicondyle and a fossa.")

    img_h, img_w = img.shape[:2]

    epic_sorted = sorted(epicondyles, key=lambda b: _box_center(b)[0])
    lat_box = epic_sorted[-1]
    lat_c = _box_center(lat_box)

    fx1, fy1, fx2, fy2 = fossa
    f_cx = (fx1 + fx2) / 2.0
    f_h = fy2 - fy1
    f_top_y = fy1

    wires = []
    wire_angles = []

    # Base direction: central wire slightly above fossa
    target_y = f_top_y - 0.2 * f_h
    if target_y < 0:
        target_y = f_top_y
    target = (f_cx, target_y)
    base_angle = math.atan2(target[1] - lat_c[1], target[0] - lat_c[0])

    offsets_deg = [-20, 0, 20]
    start_offsets_y = [6, 0, -6]

    for off_deg, dy in zip(offsets_deg, start_offsets_y):
        start = (lat_c[0], lat_c[1] + dy)
        angle = base_angle + math.radians(off_deg)
        x1, y1 = extend_line_to_bounds(start[0], start[1], angle, img_w, img_h)
        wires.append(((int(round(start[0])), int(round(start[1]))),
                      (int(round(x1)), int(round(y1)))))
        wire_angles.append(math.degrees(angle))

    # Sort wires by endpoint y: K1 distal → K3 proximal
    wires_sorted = sorted(wires, key=lambda w: w[1][1], reverse=True)
    wires = wires_sorted

    origin = lat_c
    angles_between = []
    for i in range(len(wires)):
        for j in range(i + 1, len(wires)):
            p1 = wires[i][1]
            p2 = wires[j][1]
            v1 = (p1[0] - origin[0], p1[1] - origin[1])
            v2 = (p2[0] - origin[0], p2[1] - origin[1])
            angles_between.append(compute_angle_between(v1, v2))

    info = {
        "type": "Lateral-only (3 wires)",
        "lateral_epicondyle": lat_c,
        "fossa_center": (f_cx, f_top_y),
        "wire_directions_deg": wire_angles,
        "angles_between_wires_deg": angles_between,
        "num_wires": num_wires,
    }
    return wires, info


# =========================
# SEGMENTATION VISUALIZATION
# =========================

def draw_landmark_segmentation(img, landmarks):
    """
    Semi-transparent overlays:
    - Humerus: light grey
    - Fossa: blue
    - Epicondyles: yellow
    """
    seg = img.copy()
    h, w = seg.shape[:2]
    masks = landmarks.get("masks", {})

    def apply_overlay(base, mask, color, alpha):
        overlay = base.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

    hum_mask = masks.get("humerus")
    if hum_mask is not None:
        m = hum_mask
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        seg = apply_overlay(seg, m, COL_HUMERUS, alpha=0.45)

    f_mask = masks.get("fossa")
    if f_mask is not None:
        m = f_mask
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        seg = apply_overlay(seg, m, COL_FOSSA, alpha=0.35)

    epic_masks = masks.get("epicondyles", [])
    for m in epic_masks:
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        seg = apply_overlay(seg, m, COL_EPIC, alpha=0.35)

    return seg


def draw_plan_on_image(img, landmarks, wires, labels_at_mid=False):
    """
    Draw segmentation contours + K-wires on top of original image.
    If labels_at_mid is True, K1–K3 are placed around the middle of each line
    (slightly offset perpendicular to the wire).
    Otherwise labels are placed near the entry point (used for cross formation).
    """
    out = img.copy()
    h, w = out.shape[:2]
    masks = landmarks.get("masks", {})

    # Humerus contour
    hum_mask = masks.get("humerus")
    if hum_mask is not None:
        m = hum_mask
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m8 = (m * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_HUMERUS, 2)

    # Fossa contour
    f_mask = masks.get("fossa")
    if f_mask is not None:
        m = f_mask
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m8 = (m * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_FOSSA, 2)

    # Epicondyle contours + centers
    epic_masks = masks.get("epicondyles", [])
    epic_boxes = landmarks["epicondyles"]
    for box, m in zip(epic_boxes, epic_masks):
        cx, cy = _box_center(box)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m8 = (m * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_EPIC, 2)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 4, COL_EPIC, -1)

    # K-wires + labels
    for idx, (p0, p1) in enumerate(wires, start=1):
        cv2.line(out, p0, p1, COL_WIRE, 3)

        if labels_at_mid:
            # label around the middle of the wire, offset perpendicular to the line
            mx = (p0[0] + p1[0]) / 2.0
            my = (p0[1] + p1[1]) / 2.0

            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            length = math.hypot(dx, dy)
            if length == 0:
                label_x = int(mx)
                label_y = int(my)
            else:
                nx = -dy / length
                ny = dx / length
                offset = 18
                label_x = int(mx + nx * offset)
                label_y = int(my + ny * offset)

            cv2.putText(
                out,
                f"K{idx}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COL_WIRE,
                2,
                cv2.LINE_AA,
            )
        else:
            # cross formation: label near entry point, slightly below
            offset_x = 5
            offset_y = 18 + 10 * (idx - 1)
            label_pos = (p0[0] + offset_x, p0[1] + offset_y)
            cv2.putText(
                out,
                f"K{idx}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COL_WIRE,
                2,
                cv2.LINE_AA,
            )

    return out


# =========================
# PDF REPORT
# =========================

def _wrapped_paragraph(text, width=95):
    lines = []
    for para in text.strip().split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        for line in textwrap.wrap(para, width=width):
            lines.append(line)
    return lines


def generate_pdf_report(pdf_path, raw_img_path, seg_img_path,
                        cross_img_path, lat_img_path,
                        cross_info, lat_info):
    c = pdf_canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 40

    # ----- PAGE 1: FIGURES -----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "AI-Assisted K-Wire Planning Report")

    c.setFont("Helvetica", 11)
    c.drawString(margin, height - margin - 20,
                 "Supracondylar Humerus Fracture – Cross versus Lateral K-wire Configurations")

    cell_w = (width - 3 * margin) / 2
    cell_h = 170

    # Row 1: raw + segmentation
    row1_y = height - margin - 230
    if os.path.exists(raw_img_path):
        raw_img = ImageReader(raw_img_path)
        c.drawImage(raw_img, margin, row1_y, width=cell_w, height=cell_h,
                    preserveAspectRatio=True, anchor='sw')
        c.setFont("Helvetica", 9)
        c.drawString(margin, row1_y - 12, "Figure 1. Raw AP elbow X-ray")

    if os.path.exists(seg_img_path):
        seg_img = ImageReader(seg_img_path)
        c.drawImage(seg_img, margin * 2 + cell_w, row1_y, width=cell_w, height=cell_h,
                    preserveAspectRatio=True, anchor='sw')
        c.setFont("Helvetica", 9)
        c.drawString(margin * 2 + cell_w, row1_y - 12,
                     "Figure 2. AI segmentation of humerus, fossa and epicondyles")

    # Row 2: cross vs lateral
    row2_y = row1_y - cell_h - 40
    if os.path.exists(cross_img_path):
        cross_img = ImageReader(cross_img_path)
        c.drawImage(cross_img, margin, row2_y, width=cell_w, height=cell_h,
                    preserveAspectRatio=True, anchor='sw')
        c.setFont("Helvetica", 9)
        c.drawString(margin, row2_y - 12,
                     "Figure 3. Crossed (medial + lateral) 2-wire configuration")

    if os.path.exists(lat_img_path):
        lat_img = ImageReader(lat_img_path)
        c.drawImage(lat_img, margin * 2 + cell_w, row2_y, width=cell_w, height=cell_h,
                    preserveAspectRatio=True, anchor='sw')
        c.setFont("Helvetica", 9)
        c.drawString(margin * 2 + cell_w, row2_y - 12,
                     "Figure 4. Lateral-only 3-wire configuration")

    c.showPage()

    # ----- PAGE 2: TEXT -----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "AI-Assisted K-Wire Planning Report – Details")

    text_y = height - margin - 30
    text_obj = c.beginText()
    text_obj.setTextOrigin(margin, text_y)
    text_obj.setFont("Helvetica", 11)

    cross_angle = cross_info.get("angle_between_wires_deg", None)
    lat_angles = [a for a in lat_info.get("angles_between_wires_deg", []) if a is not None]

    basics = """
This report summarizes an AI-assisted planning workflow for supracondylar humerus fracture
fixation using either crossed medial–lateral K-wires or a three-wire lateral-only construct.
The model detects the distal humerus, epicondyles and olecranon fossa on the AP radiograph
and proposes mechanically reasonable K-wire trajectories.
    """

    cross_desc = "Crossed configuration (Figure 3):\n"
    if cross_angle is not None:
        cross_desc += f"• Included angle between medial and lateral wires ≈ {cross_angle:.1f}° (target ≥ 75°).\n"
    else:
        cross_desc += "• Included angle between medial and lateral wires could not be calculated reliably.\n"
    cross_desc += (
        "• Wires enter through the medial and lateral epicondyles and converge above the olecranon fossa.\n"
        "• Provides strong resistance to varus/valgus and rotational forces but carries a risk of ulnar nerve\n"
        "  irritation or iatrogenic injury on the medial side.\n"
    )

    lat_desc = "Lateral-only configuration (Figure 4):\n"
    if lat_angles:
        lat_str = ", ".join(f"{a:.1f}°" for a in lat_angles)
        lat_desc += f"• Divergent lateral wires with measured inter-wire angles ≈ {lat_str}.\n"
    else:
        lat_desc += "• Divergent lateral wires; inter-wire angles could not be measured reliably.\n"
    lat_desc += (
        "• All wires enter from the lateral epicondyle, with one central wire directed above the olecranon fossa\n"
        "  and two additional wires fanning proximally.\n"
        "• Avoids medial dissection and ulnar nerve risk but may be less stable in high-grade fractures unless\n"
        "  sufficient divergence and bicortical purchase are achieved.\n"
    )

    principles = """
General fixation principles (for educational use only):

• Aim for anatomic reduction with stability in both coronal and sagittal planes.
• Ensure adequate separation and divergence between K-wire entry points at the cortex.
• Seek bicortical purchase in stable bone while avoiding joint penetration.
• Confirm wire position, reduction and Baumann angle under fluoroscopic control.
    """

    disclaimer = """
Disclaimer:

This report is an AI-assisted planning aid intended for use by trained orthopedic surgeons.
It does not replace clinical judgement, intraoperative fluoroscopy or local treatment protocols.
Final decisions regarding technique, wire configuration and postoperative management remain
the sole responsibility of the treating surgeon.
    """

    full_text = basics + "\n" + cross_desc + "\n" + lat_desc + "\n" + principles + "\n" + disclaimer
    for line in _wrapped_paragraph(full_text, width=95):
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()


# =========================
# TKINTER UI
# =========================

class KWirePlannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Supracondylar Humerus – K-Wire Planning")
        self.root.geometry("1500x800")
        self.root.configure(bg=BRAND_LIGHT)

        self.detector = LandmarkDetector(MODEL_PATH)

        self.image_path = None
        self.raw_cv_image = None
        self.seg_cv_image = None
        self.cross_cv_image = None
        self.lat_cv_image = None
        self.cross_info = None
        self.lat_info = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background=BRAND_LIGHT)
        style.configure("TLabel", background=BRAND_LIGHT)
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"),
                        foreground=BRAND_DARK, background=BRAND_LIGHT)
        style.configure("Subtitle.TLabel", font=("Helvetica", 11),
                        foreground="#333333", background=BRAND_LIGHT)
        style.configure("TButton", font=("Helvetica", 11, "bold"))

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        # LEFT: controls
        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 16))

        ttk.Label(left, text="AI K-Wire Planner", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(left, text="Supracondylar Humerus – AP X-ray",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(0, 24))

        upload_btn = ttk.Button(left, text="Upload X-ray", command=self.load_image)
        upload_btn.pack(fill="x", pady=(0, 8))

        run_btn = ttk.Button(left, text="Run Planning (Cross + Lateral)",
                             command=self.run_planning)
        run_btn.pack(fill="x", pady=(8, 8))

        pdf_btn = ttk.Button(left, text="Save PDF Report", command=self.save_pdf)
        pdf_btn.pack(fill="x", pady=(0, 8))

        self.info_label = ttk.Label(left, text="", style="Subtitle.TLabel",
                                    wraplength=280, justify="left")
        self.info_label.pack(fill="x", pady=(16, 0))

        # RIGHT: 3-column layout
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)

        self.raw_canvas = tk.Label(canvas_frame, bg=BRAND_WHITE, bd=1, relief="solid")
        self.seg_canvas = tk.Label(canvas_frame, bg=BRAND_WHITE, bd=1, relief="solid")
        self.cross_canvas = tk.Label(canvas_frame, bg=BRAND_WHITE, bd=1, relief="solid")
        self.lat_canvas = tk.Label(canvas_frame, bg=BRAND_WHITE, bd=1, relief="solid")

        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=1)
        canvas_frame.columnconfigure(2, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.rowconfigure(1, weight=1)

        self.raw_canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        self.seg_canvas.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=8)
        self.cross_canvas.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=(0, 4))
        self.lat_canvas.grid(row=1, column=2, sticky="nsew", padx=(8, 0), pady=(4, 0))

        captions = ttk.Frame(right)
        captions.pack(fill="x", pady=(6, 0))
        ttk.Label(captions, text="Raw X-ray", style="Subtitle.TLabel").grid(row=0, column=0, sticky="n", padx=(0, 8))
        ttk.Label(captions, text="Segmentation Map", style="Subtitle.TLabel").grid(row=0, column=1, sticky="n", padx=8)
        ttk.Label(captions, text="Cross 2-wire (top) / Lateral 3-wire (bottom)",
                  style="Subtitle.TLabel").grid(row=0, column=2, sticky="n", padx=(8, 0))
        captions.columnconfigure(0, weight=1)
        captions.columnconfigure(1, weight=1)
        captions.columnconfigure(2, weight=1)

    # -------- helpers --------

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select AP elbow X-ray",
            filetypes=[
                ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        self.image_path = path
        self.raw_cv_image = img
        self.seg_cv_image = None
        self.cross_cv_image = None
        self.lat_cv_image = None
        self.cross_info = None
        self.lat_info = None
        self.info_label.config(text="")

        self._display_on_label(self.raw_canvas, self.raw_cv_image)
        self.seg_canvas.config(image="", text="")
        self.cross_canvas.config(image="", text="")
        self.lat_canvas.config(image="", text="")

    def _display_on_label(self, label, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        label.update_idletasks()
        w = label.winfo_width() or 400
        h = label.winfo_height() or 400

        img_pil.thumbnail((w, h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_pil)
        label.img_ref = tk_img
        label.config(image=tk_img)

    def run_planning(self):
        if self.image_path is None or self.raw_cv_image is None:
            messagebox.showwarning("No image", "Please upload an AP elbow X-ray first.")
            return

        try:
            landmarks = self.detector.detect(self.image_path)
        except Exception as e:
            messagebox.showerror("Model error", f"Failed to run segmentation model:\n{e}")
            return

        if not landmarks["epicondyles"] or landmarks["fossa"] is None:
            messagebox.showerror(
                "Detection error",
                "Could not reliably detect epicondyles and/or olecranon fossa.\n"
                "Please check the image quality and orientation."
            )
            return

        # segmentation view
        self.seg_cv_image = draw_landmark_segmentation(self.raw_cv_image, landmarks)

        # Cross plan
        try:
            cross_wires, cross_info = plan_cross_formation(self.raw_cv_image, landmarks)
        except Exception as e:
            messagebox.showerror("Cross planning error", str(e))
            return
        self.cross_cv_image = draw_plan_on_image(
            self.raw_cv_image, landmarks, cross_wires, labels_at_mid=False
        )
        self.cross_info = cross_info

        # Lateral-only plan
        try:
            lat_wires, lat_info = plan_lateral_formation(self.raw_cv_image, landmarks)
        except Exception as e:
            messagebox.showerror("Lateral planning error", str(e))
            return
        self.lat_cv_image = draw_plan_on_image(
            self.raw_cv_image, landmarks, lat_wires, labels_at_mid=True
        )
        self.lat_info = lat_info

        # update previews
        self._display_on_label(self.raw_canvas, self.raw_cv_image)
        self._display_on_label(self.seg_canvas, self.seg_cv_image)
        self._display_on_label(self.cross_canvas, self.cross_cv_image)
        self._display_on_label(self.lat_canvas, self.lat_cv_image)

        # info text
        lines = []
        if self.cross_info:
            ang = self.cross_info.get("angle_between_wires_deg", None)
            if ang is not None:
                lines.append(f"Crossed 2-wire: included angle ≈ {ang:.1f}° (target ≥ 75°).")
            else:
                lines.append("Crossed 2-wire: included angle not measurable.")
        if self.lat_info:
            lat_angles = [a for a in self.lat_info.get("angles_between_wires_deg", []) if a is not None]
            if lat_angles:
                lat_str = ", ".join(f"{a:.1f}°" for a in lat_angles)
                lines.append(f"Lateral 3-wire: inter-wire angles ≈ {lat_str}.")
            else:
                lines.append("Lateral 3-wire: inter-wire angles not measurable.")

        lines.append(
            "\nNote: This is an AI-assisted educational tool. Final K-wire positions "
            "must always be confirmed intraoperatively under fluoroscopy."
        )
        self.info_label.config(text="\n".join(lines))

    def save_pdf(self):
        if (self.raw_cv_image is None or self.seg_cv_image is None or
                self.cross_cv_image is None or self.lat_cv_image is None or
                self.cross_info is None or self.lat_info is None):
            messagebox.showwarning(
                "No plan",
                "Please upload an image and run planning before saving a report."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            filetypes=[("PDF files", ("*.pdf",))]
        )
        if not path:
            return

        tmp_dir = os.path.join(os.path.dirname(path), "_kwire_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        raw_path = os.path.join(tmp_dir, "raw.png")
        seg_path = os.path.join(tmp_dir, "seg.png")
        cross_path = os.path.join(tmp_dir, "cross.png")
        lat_path = os.path.join(tmp_dir, "lat.png")

        cv2.imwrite(raw_path, self.raw_cv_image)
        cv2.imwrite(seg_path, self.seg_cv_image)
        cv2.imwrite(cross_path, self.cross_cv_image)
        cv2.imwrite(lat_path, self.lat_cv_image)

        try:
            generate_pdf_report(path, raw_path, seg_path,
                                cross_path, lat_path,
                                self.cross_info, self.lat_info)
        except Exception as e:
            messagebox.showerror("PDF error", f"Failed to generate PDF:\n{e}")
            return

        # clean up temp files
        try:
            os.remove(raw_path)
            os.remove(seg_path)
            os.remove(cross_path)
            os.remove(lat_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

        messagebox.showinfo("Saved", f"PDF report saved to:\n{path}")


# =========================
# MAIN
# =========================

def main():
    root = tk.Tk()
    app = KWirePlannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
