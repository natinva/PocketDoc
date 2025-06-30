import tkinter as tk
from tkinter import messagebox, StringVar, IntVar, Radiobutton, Frame, Label
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import math

# ---------------------------
# Global: Load your YOLOv8 segmentation model
# ---------------------------
MODEL_PATH = "/Users/avnitan/Downloads/Modeller/k-teli.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")


# ---------------------------
# Helper: Extend a line by a given ratio at both ends.
# ---------------------------
def extend_line(ptA, ptB, ratio=0.30):
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

# ---------------------------
# Helper: Compute intersection of infinite line through S->T with a segment P-Q.
# Returns (t, u, point) if intersection exists; else None.
# ---------------------------
def line_intersection_with_segment(S, T, P, Q):
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

# ---------------------------
# Helper: Extend line S->T until it hits the humerus polygon boundary.
# ---------------------------
def extend_to_humerus(S, T, humerus_polygon):
    pts = humerus_polygon.reshape(-1, 2)
    best_t = None
    best_pt = None
    # Loop over edges of the humerus polygon.
    for i in range(len(pts)):
        P = pts[i]
        Q = pts[(i + 1) % len(pts)]
        res = line_intersection_with_segment(S, T, P, Q)
        if res is not None:
            t, u, pt = res
            # We want intersections past the original endpoint (t>=1)
            if t >= 1:
                if best_t is None or t > best_t:
                    best_t = t
                    best_pt = pt
    if best_pt is None:
        return T
    else:
        return (int(best_pt[0]), int(best_pt[1]))


# ---------------------------
# Main processing function
# ---------------------------
def process_image(img_pil, formation="cross", cross_offset=0, side="lateral", pin_number=2, angle_choice=30):
    """
    Process the input PIL image and add segmentation and k-wire overlays.

    Parameters:
      formation      : "cross" or "one-sided"
      cross_offset   : For cross formation: vertical offset applied to the epicondyle centroid (-1, 0, or 1)
      side           : For one-sided: "lateral" or "medial"
      pin_number     : For one-sided: 2 or 3 pins
      angle_choice   : (Unused in new one-sided rules)

    Returns:
      annotated image (BGR) and info message (string).
    """
    # 1. Convert image to OpenCV BGR format.
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Run YOLO segmentation.
    results = model(img_np)
    result = results[0]
    if result.masks is None:
        return img_cv, "No segmentation masks found."
    masks = result.masks

    # Create an overlay for segmentation visualization.
    overlay = img_cv.copy()
    # Increase translucency (make overlay more transparent): alpha=0.1
    alpha = 0.1

    # Variables to hold segmentation information.
    epicondyle_list = []  # List of tuples: (centroid, confidence)
    fossa_polygon = None
    fossa_conf = 0.0
    humerus_polygon = None  # Store humerus polygon

    # 3. Process each segmentation polygon (explicit mapping: 0: epicondyle, 1: fossa, 2: humerus)
    for i, seg_xy in enumerate(masks.xy):
        cls_idx = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        if cls_idx == 0:
            label = "epicondyle"
        elif cls_idx == 1:
            label = "fossa"
        elif cls_idx == 2:
            label = "humerus"
        else:
            label = "unknown"
        polygon = seg_xy.astype(np.int32)

        if label == "epicondyle":
            color = (0, 255, 0)  # green
            cv2.fillPoly(overlay, [polygon], color)
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                cx, cy = int(polygon[0][0]), int(polygon[0][1])
            epicondyle_list.append(((cx, cy), conf))
        elif label == "fossa":
            color = (0, 0, 255)  # red
            cv2.fillPoly(overlay, [polygon], color)
            if conf > fossa_conf:
                fossa_conf = conf
                fossa_polygon = polygon
        elif label == "humerus":
            color = (255, 255, 255)  # white
            cv2.fillPoly(overlay, [polygon], color)
            # Store the humerus polygon (using the first one or best confidence)
            if humerus_polygon is None:
                humerus_polygon = polygon
        img_cv = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)

    info_msg = ""
    if not epicondyle_list:
        info_msg += "No epicondyle detected. "
    if fossa_polygon is None:
        info_msg += "No fossa detected. "
    if not epicondyle_list or fossa_polygon is None:
        return img_cv, info_msg

    # Ensure humerus_polygon exists; if not, wires remain unextended.
    if humerus_polygon is None:
        info_msg += "No humerus detected; wires not extended to humerus boundary. "

    if formation.lower() == "cross":
        # --- CROSS FORMATION ---
        epicondyle_list.sort(key=lambda x: x[0][0])
        left_epic = epicondyle_list[0][0]
        right_epic = epicondyle_list[-1][0]
        left_start = (left_epic[0], left_epic[1] + cross_offset)
        right_start = (right_epic[0], right_epic[1] + cross_offset)
        fossa_top = np.min(fossa_polygon[:, 1])
        intersection_x = int((left_epic[0] + right_epic[0]) / 2)
        intersection_y = int(fossa_top - 5)
        intersection_pt = (intersection_x, intersection_y)
        # Extend lines by 30% (ratio=0.30)
        left_line_start, left_line_end = extend_line(left_start, intersection_pt, ratio=0.30)
        right_line_start, right_line_end = extend_line(right_start, intersection_pt, ratio=0.30)
        # If humerus_polygon is available, extend the endpoints to humerus boundary.
        if humerus_polygon is not None:
            left_line_end = extend_to_humerus(left_start, left_line_end, humerus_polygon)
            right_line_end = extend_to_humerus(right_start, right_line_end, humerus_polygon)
        overlay_lines = img_cv.copy()
        line_color = (255, 0, 0)  # blue
        thickness = 5
        cv2.line(overlay_lines, left_line_start, left_line_end, line_color, thickness)
        cv2.line(overlay_lines, right_line_start, right_line_end, line_color, thickness)
        alpha_line = 0.8
        img_cv = cv2.addWeighted(overlay_lines, alpha_line, img_cv, 1 - alpha_line, 0)
        info_msg += f"Cross Formation: {len(epicondyle_list)} epicondyles. Fossa conf: {fossa_conf:.2f}."

    elif formation.lower() == "one-sided":
        # --- ONE-SIDED FIXATION ---
        M_fossa = cv2.moments(fossa_polygon)
        if M_fossa["m00"] != 0:
            fx = int(M_fossa["m10"]/M_fossa["m00"])
            fy = int(M_fossa["m01"]/M_fossa["m00"])
        else:
            fx, fy = int(np.mean(fossa_polygon[:,0])), int(np.mean(fossa_polygon[:,1]))
        fossa_center = (fx, fy)
        fossa_top = np.min(fossa_polygon[:, 1])
        fossa_bottom = np.max(fossa_polygon[:, 1])
        fossa_mid = int((fossa_top + fossa_bottom) / 2)
        if side.lower() == "lateral":
            selected = min(epicondyle_list, key=lambda x: x[0][0])[0]
        else:
            selected = max(epicondyle_list, key=lambda x: x[0][0])[0]
        overlay_lines = img_cv.copy()
        line_color = (255, 0, 0)  # blue
        thickness = 5

        if pin_number == 2:
            # 2-pin formation:
            target_top = (fx, int(fossa_top - 5))
            target_bottom = (fx, int(fossa_mid + 2))  # 2 px below the fossa midpoint
            start_top = (selected[0] + 1, selected[1])
            start_bottom = (selected[0] - 1, selected[1])
            s_top_ext, e_top_ext = extend_line(start_top, target_top, ratio=0.30)
            s_bot_ext, e_bot_ext = extend_line(start_bottom, target_bottom, ratio=0.30)
            if humerus_polygon is not None:
                e_top_ext = extend_to_humerus(start_top, e_top_ext, humerus_polygon)
                e_bot_ext = extend_to_humerus(start_bottom, e_bot_ext, humerus_polygon)
            cv2.line(overlay_lines, s_top_ext, e_top_ext, line_color, thickness)
            cv2.line(overlay_lines, s_bot_ext, e_bot_ext, line_color, thickness)
            info_msg += f"One-sided (2 pin, {side}): top target {target_top}, bottom target {target_bottom}."
        elif pin_number == 3:
            # 3-pin formation new logic:
            # Use the 2-pin targets as baseline.
            bottom_target = (fx, int(fossa_mid + 2))
            top_target = (fx, int(fossa_top - 5))
            # The middle target is the midpoint between bottom_target and top_target.
            mid_target = (fx, int((bottom_target[1] + top_target[1]) / 2))
            start_bottom = (selected[0] - 1, selected[1])
            start_mid = selected
            start_top = (selected[0] + 1, selected[1])
            s_bot_ext, e_bot_ext = extend_line(start_bottom, bottom_target, ratio=0.30)
            s_mid_ext, e_mid_ext = extend_line(start_mid, mid_target, ratio=0.30)
            s_top_ext, e_top_ext = extend_line(start_top, top_target, ratio=0.30)
            if humerus_polygon is not None:
                e_bot_ext = extend_to_humerus(start_bottom, e_bot_ext, humerus_polygon)
                e_mid_ext = extend_to_humerus(start_mid, e_mid_ext, humerus_polygon)
                e_top_ext = extend_to_humerus(start_top, e_top_ext, humerus_polygon)
            cv2.line(overlay_lines, s_bot_ext, e_bot_ext, line_color, thickness)
            cv2.line(overlay_lines, s_mid_ext, e_mid_ext, line_color, thickness)
            cv2.line(overlay_lines, s_top_ext, e_top_ext, line_color, thickness)
            info_msg += (f"One-sided (3 pin, {side}): targets - bottom {bottom_target}, mid {mid_target}, top {top_target}.")
        alpha_line = 0.8
        img_cv = cv2.addWeighted(overlay_lines, alpha_line, img_cv, 1 - alpha_line, 0)
        info_msg += f" Fossa conf: {fossa_conf:.2f}."

    return img_cv, info_msg


# ---------------------------
# Tkinter Live Inference Application with Controls
# ---------------------------
class YOLOSegmentationLiveApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Live Inference with K-Wire Options")
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

        # Control Panel Frame
        control_frame = Frame(window)
        control_frame.pack(pady=5)

        # Formation Type Radiobuttons
        self.formation = StringVar(value="cross")
        Label(control_frame, text="Formation:").grid(row=0, column=0, padx=5)
        Radiobutton(control_frame, text="Cross Formation", variable=self.formation, value="cross").grid(row=0, column=1)
        Radiobutton(control_frame, text="One-Sided Fixation", variable=self.formation, value="one-sided").grid(row=0, column=2)

        # For Cross Formation: Starting Point Level as Radiobuttons
        Label(control_frame, text="Cross Starting Level:").grid(row=1, column=0, padx=5)
        self.cross_offset = IntVar(value=0)
        Radiobutton(control_frame, text="1 px above", variable=self.cross_offset, value=-1).grid(row=1, column=1)
        Radiobutton(control_frame, text="Neutral", variable=self.cross_offset, value=0).grid(row=1, column=2)
        Radiobutton(control_frame, text="1 px below", variable=self.cross_offset, value=1).grid(row=1, column=3)

        # For One-Sided Fixation: Side Selection
        Label(control_frame, text="Side (One-Sided):").grid(row=2, column=0, padx=5)
        self.side = StringVar(value="lateral")
        Radiobutton(control_frame, text="Lateral", variable=self.side, value="lateral").grid(row=2, column=1)
        Radiobutton(control_frame, text="Medial", variable=self.side, value="medial").grid(row=2, column=2)

        # For One-Sided Fixation: Pin Number
        Label(control_frame, text="Pins (One-Sided):").grid(row=3, column=0, padx=5)
        self.pin_number = IntVar(value=2)
        Radiobutton(control_frame, text="2 Pins", variable=self.pin_number, value=2).grid(row=3, column=1)
        Radiobutton(control_frame, text="3 Pins", variable=self.pin_number, value=3).grid(row=3, column=2)

        # For One-Sided with 2 Pins: Angle Selection as Radiobuttons
        Label(control_frame, text="Angle (2 Pins):").grid(row=4, column=0, padx=5)
        self.angle_choice = IntVar(value=30)
        Radiobutton(control_frame, text="30°", variable=self.angle_choice, value=30).grid(row=4, column=1)
        Radiobutton(control_frame, text="45°", variable=self.angle_choice, value=45).grid(row=4, column=2)
        Radiobutton(control_frame, text="60°", variable=self.angle_choice, value=60).grid(row=4, column=3)

        # Info Label and Quit Button
        self.info_label = Label(window, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)
        self.btn_quit = tk.Button(window, text="Quit", command=self.on_closing, font=("Arial", 12))
        self.btn_quit.pack(pady=5)

        # Zoom Slider
        Label(control_frame, text="Zoom Level:").grid(row=5, column=0, padx=5)
        self.zoom_factor = tk.DoubleVar(value=1.2)
        zoom_slider = tk.Scale(
            control_frame,
            from_=1.2,
            to=7.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.zoom_factor,
            label="Zoom",
            length=200
        )
        zoom_slider.grid(row=5, column=1, columnspan=3, sticky="we", padx=5)

        # Manual Zoom Entry
        Label(control_frame, text="Manual Zoom:").grid(row=6, column=0, padx=5)
        self.zoom_entry = tk.Entry(control_frame, width=6)
        self.zoom_entry.grid(row=6, column=1)
        self.zoom_entry.bind("<Return>", self.set_manual_zoom)

        # Quality Selector
        Label(control_frame, text="Inference Quality:").grid(row=7, column=0, padx=5)
        self.quality = StringVar(value="normal")
        Radiobutton(control_frame, text="Best", variable=self.quality, value="best").grid(row=7, column=1)
        Radiobutton(control_frame, text="Good", variable=self.quality, value="good").grid(row=7, column=2)
        Radiobutton(control_frame, text="Normal", variable=self.quality, value="normal").grid(row=7, column=3)
        Radiobutton(control_frame, text="Low", variable=self.quality, value="low").grid(row=7, column=4)

        self.delay = 15  # update delay in ms
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Retrieve the current zoom factor from the slider
            zoom_factor = self.zoom_factor.get()
            h, w, _ = frame.shape
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            zoomed_frame = frame[y1:y1 + new_h, x1:x1 + new_w]

            # Resize for faster inference
            quality = self.quality.get()
            if quality == "low":
                inference_width = 320
            elif quality == "normal":
                inference_width = 480
            elif quality == "good":
                inference_width = 640
            elif quality == "best":
                inference_width = w  # full original camera width
            else:
                inference_width = 480  # fallback
            inference_height = int(inference_width * new_h / new_w)
            resized_for_inference = cv2.resize(zoomed_frame, (inference_width, inference_height))

            # Convert to PIL for processing
            frame_rgb = cv2.cvtColor(resized_for_inference, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            # Run model and draw overlays
            annotated_frame, info_msg = process_image(
                img_pil,
                formation=self.formation.get(),
                cross_offset=self.cross_offset.get(),
                side=self.side.get(),
                pin_number=self.pin_number.get(),
                angle_choice=self.angle_choice.get()
            )

            # Resize for display to fit window/canvas
            display_width = 640
            display_height = int(display_width * annotated_frame.shape[0] / annotated_frame.shape[1])
            annotated_resized = cv2.resize(annotated_frame, (display_width, display_height))

            annotated_rgb = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)
            imgtk = ImageTk.PhotoImage(image=annotated_pil)

            self.canvas.config(width=display_width, height=display_height)
            self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
            self.canvas.imgtk = imgtk  # prevent garbage collection
            self.info_label.config(text=info_msg)

        self.window.after(self.delay, self.update)

    def set_manual_zoom(self, event=None):
        try:
            value = float(self.zoom_entry.get())
            if 1.2 <= value <= 7.0:
                self.zoom_factor.set(value)
            else:
                messagebox.showwarning("Invalid Zoom", "Enter a value between 1.2 and 7.0")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a numeric zoom value")

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