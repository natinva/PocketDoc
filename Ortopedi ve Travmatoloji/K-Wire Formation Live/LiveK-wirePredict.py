import tkinter as tk
from tkinter import messagebox, StringVar, IntVar, Radiobutton, Frame, Label
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
from pathlib import Path
import json
from datetime import datetime
import torch
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------------------------
# Model Loading
# ---------------------------

MODEL_PATH = Path(
    "/Users/avnitan/PycharmProjects/TestProject/PocketDoc/Modeller/"
    "Orthopaedics and Traumatology/Supracondylar Humerus - AP Xray/k-teli.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")

try:
    model = YOLO(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")


# ---------------------------
# Geometry Helpers
# ---------------------------

def extend_line(ptA, ptB, ratio=0.30):
    """Extend a line segment AB by 'ratio' of its length at both ends."""
    A = np.array(ptA, dtype=np.float32)
    B = np.array(ptB, dtype=np.float32)
    v = B - A
    norm = np.linalg.norm(v)
    if norm == 0:
        return ptA, ptB
    unit = v / norm
    newA = A - unit * (ratio * norm)
    newB = B + unit * (ratio * norm)
    return (int(newA[0]), int(newA[1])), (int(newB[0]), int(newB[1]))


def line_intersection_with_segment(S, T, P, Q):
    """
    Intersection of infinite line through S->T with segment P-Q.
    Returns (t, u, point) if intersection exists; else None.
    """
    S = np.array(S, dtype=np.float32)
    T = np.array(T, dtype=np.float32)
    P = np.array(P, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    v = T - S
    w = Q - P
    denom = v[0] * w[1] - v[1] * w[0]
    if denom == 0:
        return None
    diff = P - S
    t = (diff[0] * w[1] - diff[1] * w[0]) / denom
    u = (diff[0] * v[1] - diff[1] * v[0]) / denom
    if u < 0 or u > 1:
        return None
    intersection = S + t * v
    return t, u, (intersection[0], intersection[1])


def extend_to_humerus(S, T, humerus_polygon):
    """
    Extend line S->T until it hits the humerus polygon boundary.
    """
    pts = humerus_polygon.reshape(-1, 2)
    best_t = None
    best_pt = None
    for i in range(len(pts)):
        P = pts[i]
        Q = pts[(i + 1) % len(pts)]
        res = line_intersection_with_segment(S, T, P, Q)
        if res is not None:
            t, u, pt = res
            if t >= 1:
                if best_t is None or t > best_t:
                    best_t = t
                    best_pt = pt
    if best_pt is None:
        return T
    else:
        return (int(best_pt[0]), int(best_pt[1]))


def angle_deg(p1, p2, p3, p4):
    """Angle in degrees between line p1-p2 and line p3-p4."""
    v1 = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    v2 = np.array(p4, dtype=float) - np.array(p3, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def infinite_line_intersection(p1, p2, p3, p4):
    """Intersection of two infinite lines (p1-p2) and (p3-p4)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (float(px), float(py))


def compute_humerus_bounds(humerus_polygon):
    if humerus_polygon is None:
        return None
    pts = humerus_polygon.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return {
        "min_x": float(xs.min()),
        "max_x": float(xs.max()),
        "min_y": float(ys.min()),
        "max_y": float(ys.max()),
        "height": float(ys.max() - ys.min()),
        "width": float(xs.max() - xs.min())
    }


# ---------------------------
# Pin Planning and Metrics
# ---------------------------

def plan_pins(epicondyle_list,
              fossa_polygon,
              humerus_polygon,
              formation="cross",
              side="lateral",
              pin_number=2,
              start_offset=0,
              cross_extra_side="left",
              fossa_conf=0.0):
    """
    Returns:
      wires: list of dicts {start, end, role}
      info_msg: string for UI (no warnings here)
      metrics: dict with geometry + warnings (for report)
    """

    wires = []
    info_parts = []
    metrics = {
        "formation": formation,
        "side": side,
        "pin_number": pin_number,
        "fossa_confidence": float(fossa_conf),
        "divergence_angle_deg": None,
        "crossing_height_ratio": None,
        "entry_spread_ratio": None,
        "warnings": []
    }

    if not epicondyle_list:
        info_parts.append("No epicondyle detected.")
        metrics["warnings"].append("no_epicondyle")
        return wires, " ".join(info_parts), metrics

    if fossa_polygon is None:
        info_parts.append("No fossa detected.")
        metrics["warnings"].append("no_fossa")
        return wires, " ".join(info_parts), metrics

    hum_bounds = compute_humerus_bounds(humerus_polygon) if humerus_polygon is not None else None

    fossa_top = int(np.min(fossa_polygon[:, 1]))
    fossa_bottom = int(np.max(fossa_polygon[:, 1]))
    fossa_height = max(1, fossa_bottom - fossa_top)
    # vertical spacing for higher pins – x2 again (0.80 of fossa height, min 12 px)
    delta_y = max(12, int(0.60 * fossa_height))

    # --- CROSS FORMATION ---
    if formation.lower() == "cross":
        if len(epicondyle_list) < 2:
            info_parts.append("Cross formation needs ≥2 epicondyles.")
            metrics["warnings"].append("insufficient_epicondyles_for_cross")
            return wires, " ".join(info_parts), metrics

        epicondyle_list.sort(key=lambda x: x[0][0])
        left_epic = epicondyle_list[0][0]
        right_epic = epicondyle_list[-1][0]

        left_start = (left_epic[0], left_epic[1] + start_offset)
        right_start = (right_epic[0], right_epic[1] + start_offset)

        # crossing point at EXACT top of fossa
        mid_x = int((left_epic[0] + right_epic[0]) / 2)
        base_y = fossa_top
        intersection_base = (mid_x, base_y)
        intersection_high = (mid_x, base_y - delta_y)  # extra wire above

        # Base cross wires (both cross at base_y = fossa_top)
        left_line_start, left_line_end = extend_line(left_start, intersection_base, ratio=0.30)
        right_line_start, right_line_end = extend_line(right_start, intersection_base, ratio=0.30)

        if humerus_polygon is not None:
            left_line_end = extend_to_humerus(left_start, left_line_end, humerus_polygon)
            right_line_end = extend_to_humerus(right_start, right_line_end, humerus_polygon)

        wires.append({"start": left_line_start, "end": left_line_end, "role": "cross_left"})
        wires.append({"start": right_line_start, "end": right_line_end, "role": "cross_right"})

        # 3rd cross wire crossing ABOVE top of fossa on chosen side
        if pin_number == 3:
            extra_side = cross_extra_side.lower()
            if extra_side == "left":
                base_start = left_start
            else:
                base_start = right_start

            extra_start, extra_end = extend_line(base_start, intersection_high, ratio=0.30)
            if humerus_polygon is not None:
                extra_end = extend_to_humerus(base_start, extra_end, humerus_polygon)

            wires.append({
                "start": extra_start,
                "end": extra_end,
                "role": f"cross_extra_{extra_side}"
            })

        info_parts.append(
            f"Cross Formation ({pin_number} wires). Fossa conf: {fossa_conf:.2f}."
        )

        # Metrics
        if len(wires) >= 2:
            ang = angle_deg(wires[0]["start"], wires[0]["end"],
                            wires[1]["start"], wires[1]["end"])
            metrics["divergence_angle_deg"] = ang

            inter = infinite_line_intersection(
                wires[0]["start"], wires[0]["end"],
                wires[1]["start"], wires[1]["end"]
            )
            if inter is not None and hum_bounds is not None and hum_bounds["height"] > 0:
                crossing_y = inter[1]
                ratio = (hum_bounds["max_y"] - crossing_y) / hum_bounds["height"]
                metrics["crossing_height_ratio"] = float(ratio)

        if hum_bounds is not None and hum_bounds["width"] > 0:
            entry_distance = float(abs(left_epic[0] - right_epic[0]))
            metrics["entry_spread_ratio"] = entry_distance / hum_bounds["width"]

    # --- ONE-SIDED FIXATION ---
    elif formation.lower() == "one-sided":
        M_fossa = cv2.moments(fossa_polygon)
        if M_fossa["m00"] != 0:
            fx = int(M_fossa["m10"] / M_fossa["m00"])
            fy = int(M_fossa["m01"] / M_fossa["m00"])
        else:
            fx = int(np.mean(fossa_polygon[:, 0]))
            fy = int(np.mean(fossa_polygon[:, 1]))

        # epicondyle entry: lateral vs medial (mirror safe, only affects x)
        if side.lower() == "lateral":
            selected = max(epicondyle_list, key=lambda x: x[0][0])[0]
        else:  # medial
            selected = min(epicondyle_list, key=lambda x: x[0][0])[0]

        start = (selected[0], selected[1] + start_offset)

        # All wires cross x = fx at or above fossa_top
        if pin_number == 2:
            target_ys = [fossa_top, fossa_top - delta_y]
            roles = ["one_sided_bottom", "one_sided_top"]
        else:  # 3 pins
            target_ys = [fossa_top, fossa_top - delta_y, fossa_top - 2 * delta_y]
            roles = ["one_sided_bottom", "one_sided_mid", "one_sided_top"]

        wires_local = []
        for ty, role in zip(target_ys, roles):
            target = (fx, ty)
            s_ext, e_ext = extend_line(start, target, ratio=0.30)
            if humerus_polygon is not None:
                e_ext = extend_to_humerus(start, e_ext, humerus_polygon)
            wires_local.append({"start": s_ext, "end": e_ext, "role": role})

        wires.extend(wires_local)

        info_parts.append(
            f"One-sided ({side}, {pin_number} pins). Fossa conf: {fossa_conf:.2f}."
        )

        # Metrics
        if len(wires_local) >= 2:
            ang = angle_deg(wires_local[0]["start"], wires_local[0]["end"],
                            wires_local[1]["start"], wires_local[1]["end"])
            metrics["divergence_angle_deg"] = ang

        if hum_bounds is not None and hum_bounds["width"] > 0:
            xs = [w["start"][0] for w in wires_local]
            if len(xs) > 1:
                entry_distance = float(max(xs) - min(xs))
            else:
                entry_distance = 0.0
            metrics["entry_spread_ratio"] = entry_distance / hum_bounds["width"]

    # Warnings for report
    if metrics["fossa_confidence"] < 0.5:
        metrics["warnings"].append("low_fossa_confidence")

    if metrics["divergence_angle_deg"] is not None and metrics["divergence_angle_deg"] < 30:
        metrics["warnings"].append("low_divergence_angle")

    return wires, " ".join(info_parts), metrics


# ---------------------------
# Main Image Processing
# ---------------------------

def process_image(img_pil,
                  formation="cross",
                  start_offset=0,
                  side="lateral",
                  pin_number=2,
                  cross_extra_side="left"):
    """
    Returns:
      annotated BGR image,
      info_msg (str),
      metrics (dict)
    """
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        results = model(img_np)

    result = results[0]
    if result.masks is None:
        return img_cv, "No segmentation masks found.", {}

    masks = result.masks
    overlay = img_cv.copy()
    alpha = 0.1

    epicondyle_list = []   # (centroid, conf)
    fossa_polygon = None
    fossa_conf = 0.0
    humerus_polygon = None

    for i, seg_xy in enumerate(masks.xy):
        cls_idx = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])

        if cls_idx == 0:
            label = "epicondyle"
            color = (0, 255, 0)
        elif cls_idx == 1:
            label = "fossa"
            color = (0, 0, 255)
        elif cls_idx == 2:
            label = "humerus"
            color = (255, 255, 255)
        else:
            label = "unknown"
            color = (128, 128, 128)

        polygon = seg_xy.astype(np.int32)

        if label == "epicondyle":
            cv2.fillPoly(overlay, [polygon], color)
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(polygon[0][0]), int(polygon[0][1])
            epicondyle_list.append(((cx, cy), conf))

        elif label == "fossa":
            cv2.fillPoly(overlay, [polygon], color)
            if conf > fossa_conf:
                fossa_conf = conf
                fossa_polygon = polygon

        elif label == "humerus":
            cv2.fillPoly(overlay, [polygon], color)
            if humerus_polygon is None:
                humerus_polygon = polygon

    img_cv = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)

    if not epicondyle_list or fossa_polygon is None:
        info_msg = ""
        if not epicondyle_list:
            info_msg += "No epicondyle detected. "
        if fossa_polygon is None:
            info_msg += "No fossa detected. "
        if humerus_polygon is None:
            info_msg += "No humerus detected. "
        return img_cv, info_msg, {}

    if humerus_polygon is None:
        cortex_info = "No humerus detected; wires not extended to cortex. "
    else:
        cortex_info = ""

    wires, info_msg, metrics = plan_pins(
        epicondyle_list=epicondyle_list,
        fossa_polygon=fossa_polygon,
        humerus_polygon=humerus_polygon,
        formation=formation,
        side=side,
        pin_number=pin_number,
        start_offset=start_offset,
        cross_extra_side=cross_extra_side,
        fossa_conf=fossa_conf
    )

    info_msg = cortex_info + info_msg

    overlay_lines = img_cv.copy()
    line_color = (255, 0, 0)  # blue-ish in BGR
    thickness = 5

    for w in wires:
        cv2.line(overlay_lines, w["start"], w["end"], line_color, thickness)

    img_cv = cv2.addWeighted(overlay_lines, 0.8, img_cv, 0.2, 0)

    extra = []
    if metrics.get("divergence_angle_deg") is not None:
        extra.append(f"Divergence: {metrics['divergence_angle_deg']:.1f}°")
    if metrics.get("crossing_height_ratio") is not None:
        extra.append(f"Crossing height (rel): {metrics['crossing_height_ratio']:.2f}")
    if metrics.get("entry_spread_ratio") is not None:
        extra.append(f"Entry spread (rel): {metrics['entry_spread_ratio']:.2f}")
    if extra:
        info_msg += " | " + " | ".join(extra)

    return img_cv, info_msg, metrics


# ---------------------------
# Report Helpers
# ---------------------------

def build_plan_summary(name, plan):
    metrics = plan["metrics"]
    lines = [f"{name}"]
    div = metrics.get("divergence_angle_deg")
    spread = metrics.get("entry_spread_ratio")
    cross_h = metrics.get("crossing_height_ratio")
    warnings = metrics.get("warnings", [])

    pros = []
    cons = []

    if div is not None:
        lines.append(f"- Divergence angle: {div:.1f}°")
        if div >= 30:
            pros.append("Good divergence (≥30°) – likely stable construct.")
        else:
            cons.append("Low divergence (<30°) – potential mechanical weakness.")

    if spread is not None:
        lines.append(f"- Entry spread (rel. humerus width): {spread:.2f}")
        if spread >= 0.25:
            pros.append("Adequate lateral spread of entry points.")
        else:
            cons.append("Narrow entry spread – reduced buttressing.")

    if cross_h is not None:
        lines.append(f"- Crossing height (relative): {cross_h:.2f}")

    if warnings:
        lines.append(f"- Warnings: {', '.join(warnings)}")

    if pros:
        lines.append("Pros:")
        for p in pros:
            lines.append(f"• {p}")

    if cons:
        lines.append("Cons:")
        for c in cons:
            lines.append(f"• {c}")

    return lines


def draw_image_centered(c, img_path, page_width, page_height,
                        top_margin=70, bottom_margin=80):
    img = ImageReader(str(img_path))
    iw, ih = img.getSize()
    max_w = page_width - 80
    max_h = page_height - top_margin - bottom_margin
    scale = min(max_w / iw, max_h / ih)
    w_draw = iw * scale
    h_draw = ih * scale
    x = (page_width - w_draw) / 2
    y = page_height - top_margin - h_draw
    c.drawImage(img, x, y, width=w_draw, height=h_draw)
    return y - 20  # y position under image


# ---------------------------
# Tkinter Live Inference Application
# ---------------------------

class YOLOSegmentationLiveApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Supracondylar Pin Planner – Live Inference")
        self.video_source = video_source

        # Open video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            messagebox.showerror("Video Error", "Unable to open video source")
            self.window.destroy()
            return

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # --- Control Panel ---
        control_frame = Frame(window)
        control_frame.pack(pady=5)

        # Formation
        self.formation = StringVar(value="cross")
        Label(control_frame, text="Formation:").grid(row=0, column=0, padx=5)
        Radiobutton(control_frame, text="Cross Formation",
                    variable=self.formation, value="cross",
                    command=self.update_mode_ui).grid(row=0, column=1)
        Radiobutton(control_frame, text="One-Sided Fixation",
                    variable=self.formation, value="one-sided",
                    command=self.update_mode_ui).grid(row=0, column=2)

        # Cross-specific frame
        self.cross_frame = Frame(control_frame)
        Label(self.cross_frame, text="Cross pins (2 or 3):").grid(row=0, column=0, padx=5)
        self.pin_number_cross = IntVar(value=3)  # default 3 for cross
        Radiobutton(self.cross_frame, text="2 Pins",
                    variable=self.pin_number_cross, value=2).grid(row=0, column=1)
        Radiobutton(self.cross_frame, text="3 Pins",
                    variable=self.pin_number_cross, value=3).grid(row=0, column=2)

        Label(self.cross_frame, text="Cross extra pin side:").grid(row=1, column=0, padx=5)
        self.cross_extra_side = StringVar(value="left")
        Radiobutton(self.cross_frame, text="Left",
                    variable=self.cross_extra_side, value="left").grid(row=1, column=1)
        Radiobutton(self.cross_frame, text="Right",
                    variable=self.cross_extra_side, value="right").grid(row=1, column=2)

        # One-sided-specific frame
        self.one_frame = Frame(control_frame)
        Label(self.one_frame, text="One-Sided Side:").grid(row=0, column=0, padx=5)
        self.side = StringVar(value="lateral")
        Radiobutton(self.one_frame, text="Lateral",
                    variable=self.side, value="lateral").grid(row=0, column=1)
        Radiobutton(self.one_frame, text="Medial",
                    variable=self.side, value="medial").grid(row=0, column=2)

        Label(self.one_frame, text="One-Sided pins (2 or 3):").grid(row=1, column=0, padx=5)
        self.pin_number_one = IntVar(value=3)
        Radiobutton(self.one_frame, text="2 Pins",
                    variable=self.pin_number_one, value=2).grid(row=1, column=1)
        Radiobutton(self.one_frame, text="3 Pins",
                    variable=self.pin_number_one, value=3).grid(row=1, column=2)

        # Entry level (common)
        Label(control_frame, text="Entry level (from epicondyle):").grid(row=3, column=0, padx=5)
        self.start_offset = IntVar(value=0)
        Radiobutton(control_frame, text="2 px above",
                    variable=self.start_offset, value=-2).grid(row=3, column=1)
        Radiobutton(control_frame, text="Neutral",
                    variable=self.start_offset, value=0).grid(row=3, column=2)
        Radiobutton(control_frame, text="2 px below",
                    variable=self.start_offset, value=2).grid(row=3, column=3)

        # Info label
        self.info_label = Label(window, text="", font=("Arial", 11))
        self.info_label.pack(pady=5)

        # Action buttons
        action_frame = Frame(window)
        action_frame.pack(pady=5)

        self.frozen = False
        self.freeze_button = tk.Button(action_frame, text="Freeze",
                                       command=self.toggle_freeze, font=("Arial", 11))
        self.freeze_button.grid(row=0, column=0, padx=5)

        self.capture_button = tk.Button(action_frame, text="Capture & Save",
                                        command=self.capture_and_save, font=("Arial", 11))
        self.capture_button.grid(row=0, column=1, padx=5)

        self.report_button = tk.Button(action_frame, text="Report",
                                       command=self.generate_report, font=("Arial", 11))
        self.report_button.grid(row=0, column=2, padx=5)

        self.btn_quit = tk.Button(action_frame, text="Quit",
                                  command=self.on_closing, font=("Arial", 11))
        self.btn_quit.grid(row=0, column=3, padx=5)

        # Last outputs
        self.last_annotated_bgr = None
        self.last_metrics = None
        self.last_info_msg = ""
        self.last_base_pil = None  # for report

        # Inference stride
        self.inference_stride = 2
        self.frame_counter = 0

        self.delay = 15  # ms
        self.update_mode_ui()
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_mode_ui(self):
        if self.formation.get() == "cross":
            self.cross_frame.grid(row=1, column=0, columnspan=4, pady=2, sticky="w")
            self.one_frame.grid_forget()
        else:
            self.one_frame.grid(row=1, column=0, columnspan=4, pady=2, sticky="w")
            self.cross_frame.grid_forget()

    def update(self):
        if self.frozen and self.last_annotated_bgr is not None:
            self.draw_frame(self.last_annotated_bgr, self.last_info_msg)
            self.window.after(self.delay, self.update)
            return

        ret, frame = self.vid.read()
        if not ret:
            self.window.after(self.delay, self.update)
            return

        h, w, _ = frame.shape
        inference_width = 640
        inference_height = int(inference_width * h / w)
        resized_for_inference = cv2.resize(frame, (inference_width, inference_height))

        self.frame_counter = (self.frame_counter + 1) % self.inference_stride
        if self.frame_counter == 0 or self.last_annotated_bgr is None:
            frame_rgb = cv2.cvtColor(resized_for_inference, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            self.last_base_pil = img_pil

            if self.formation.get() == "cross":
                pin_number = self.pin_number_cross.get()
                side = "lateral"  # unused in cross
                cross_side = self.cross_extra_side.get()
            else:
                pin_number = self.pin_number_one.get()
                side = self.side.get()
                cross_side = "left"

            annotated_frame, info_msg, metrics = process_image(
                img_pil,
                formation=self.formation.get(),
                start_offset=self.start_offset.get(),
                side=side,
                pin_number=pin_number,
                cross_extra_side=cross_side
            )

            self.last_annotated_bgr = annotated_frame
            self.last_metrics = metrics
            self.last_info_msg = info_msg

        if self.last_annotated_bgr is not None:
            self.draw_frame(self.last_annotated_bgr, self.last_info_msg)

        self.window.after(self.delay, self.update)

    def draw_frame(self, bgr_image, info_msg):
        display_width = 640
        display_height = int(display_width * bgr_image.shape[0] / bgr_image.shape[1])
        annotated_resized = cv2.resize(bgr_image, (display_width, display_height))

        annotated_rgb = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)
        imgtk = ImageTk.PhotoImage(image=annotated_pil)

        self.canvas.config(width=display_width, height=display_height)
        self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
        self.canvas.imgtk = imgtk
        self.info_label.config(text=info_msg)

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self.freeze_button.config(text="Unfreeze" if self.frozen else "Freeze")

    def capture_and_save(self):
        if self.last_annotated_bgr is None:
            messagebox.showwarning("No Frame", "No annotated frame available to save.")
            return

        out_dir = Path(__file__).resolve().parent / "captures"
        out_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = out_dir / f"supracondylar_plan_{timestamp}.png"
        json_path = out_dir / f"supracondylar_plan_{timestamp}.json"

        cv2.imwrite(str(img_path), self.last_annotated_bgr)

        data = {
            "timestamp": timestamp,
            "info_msg": self.last_info_msg,
            "metrics": self.last_metrics
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        messagebox.showinfo("Saved", f"Saved image and metrics:\n{img_path.name}\n{json_path.name}")

    def generate_report(self):
        if self.last_base_pil is None:
            messagebox.showwarning("No Frame", "No base frame available for report.")
            return

        img_pil = self.last_base_pil
        out_dir = Path(__file__).resolve().parent / "captures"
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = out_dir / f"supracondylar_report_{timestamp}.pdf"

        # Save original image
        orig_np = np.array(img_pil)
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        orig_path = out_dir / f"supracondylar_original_{timestamp}.png"
        cv2.imwrite(str(orig_path), orig_bgr)

        plans = {}
        img_paths = {}

        def run_plan(key, formation, pin_number, side, cross_extra_side="left"):
            annotated, _, metrics = process_image(
                img_pil,
                formation=formation,
                start_offset=self.start_offset.get(),
                side=side,
                pin_number=pin_number,
                cross_extra_side=cross_extra_side
            )
            png_path = out_dir / f"{key.replace(' ', '_').replace('–','-')}_{timestamp}.png"
            cv2.imwrite(str(png_path), annotated)
            img_paths[key] = png_path
            plans[key] = {
                "formation": formation,
                "pin_number": pin_number,
                "side": side,
                "cross_extra_side": cross_extra_side,
                "metrics": metrics
            }

        # Required plans:
        # - Cross 2 & 3
        # - Lateral 2 & 3
        # - Medial 2 & 3
        run_plan("Cross – 2 wires", "cross", 2, side="lateral",
                 cross_extra_side=self.cross_extra_side.get())
        run_plan("Cross – 3 wires", "cross", 3, side="lateral",
                 cross_extra_side=self.cross_extra_side.get())
        run_plan("Lateral – 2 wires", "one-sided", 2, side="lateral")
        run_plan("Lateral – 3 wires", "one-sided", 3, side="lateral")
        run_plan("Medial – 2 wires", "one-sided", 2, side="medial")
        run_plan("Medial – 3 wires", "one-sided", 3, side="medial")

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        margin = 40

        # Page 1 – Original image
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, height - margin, "Supracondylar Humerus K-wire Planning Report")
        c.setFont("Helvetica", 10)
        c.drawString(margin, height - margin - 20, f"Generated: {timestamp}")
        y = draw_image_centered(c, orig_path, width, height)
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, "Original AP radiograph (unprocessed).")
        c.showPage()

        # One page per plan: title + image + summary text
        for name, plan in plans.items():
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, height - margin, name)
            y = draw_image_centered(c, img_paths[name], width, height)
            c.setFont("Helvetica", 10)
            summary_lines = build_plan_summary(name, plan)
            for line in summary_lines:
                if y < margin + 20:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(margin, height - margin, name + " (cont.)")
                    y = height - margin - 30
                    c.setFont("Helvetica", 10)
                c.drawString(margin, y, line)
                y -= 14
            c.showPage()

        c.save()
        messagebox.showinfo("Report", f"Report generated:\n{pdf_path.name}")

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = YOLOSegmentationLiveApp(root, video_source=0)
    root.mainloop()
