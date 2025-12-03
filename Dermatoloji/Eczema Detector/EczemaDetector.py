import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import datetime
import os
import tempfile

from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ====== CONFIG ======
MODEL_PATH = "/Users/avnitan/PycharmProjects/TestProject/PocketDoc/Modeller/Dermatology/Eczema/eczema.pt"
CAM_W, CAM_H = 640, 480          # camera resolution (for speed)
INFERENCE_INTERVAL = 8           # run YOLO every N frames


def load_eczema_model(path: str):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def analyze_with_yolo(model, frame_bgr):
    """
    Run YOLO on BGR frame and return:
      has_eczema (bool),
      best_conf (float),
      risk_label (str),
      annotated_bgr (np.ndarray BGR),
      eczema_box_count (int)

    Boxes drawn manually, always labelled 'eczema'.
    """
    if model is None:
        return False, None, "Model not loaded", frame_bgr, 0

    h, w, _ = frame_bgr.shape
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, verbose=False)
    res = results[0]
    boxes = res.boxes

    annotated_bgr = frame_bgr.copy()
    best_conf = 0.0
    eczema_box_count = 0

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0

            # clip coords
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            eczema_box_count += 1
            if conf > best_conf:
                best_conf = conf

            label_text = f"eczema {conf * 100:.0f}%"

            # box
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # label
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x1
            text_y = max(y1 - 5, th + 2)
            cv2.rectangle(
                annotated_bgr,
                (text_x, text_y - th - 2),
                (text_x + tw + 2, text_y + baseline),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated_bgr,
                label_text,
                (text_x + 1, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    has_eczema = eczema_box_count > 0

    # Confidence logic:
    # <20% very low / uncertain
    # 20–30% doubtful
    # 30–50% probably
    # >50% eczema (high confidence)
    if not has_eczema:
        risk_label = "No clear eczema sign in this frame."
        best_conf_used = 0.0
    else:
        best_conf_used = best_conf
        if best_conf < 0.20:
            risk_label = "Very low confidence – uncertain eczema-like lesion."
        elif best_conf < 0.30:
            risk_label = "Doubtful eczema-like lesion."
        elif best_conf < 0.50:
            risk_label = "Probably eczema-like lesion."
        else:
            risk_label = "Eczema-like lesion with high confidence."

    return has_eczema, best_conf_used, risk_label, annotated_bgr, eczema_box_count


class EczemaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PocketDoc – Eczema AI Assistant (Live)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Model ---
        self.model = load_eczema_model(MODEL_PATH)
        if self.model is None:
            messagebox.showwarning(
                "Model error",
                "Could not load eczema model.\nCheck MODEL_PATH and dependencies."
            )

        # --- Camera ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Cannot open webcam.")
            self.cap = None
        else:
            # drop resolution for speed
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

        # ---- State ----
        self.current_frame_bgr = None
        self.annotated_frame_pil = None

        self.last_has_eczema = None
        self.last_confidence = None
        self.last_risk_label = None
        self.last_box_count = 0

        self.frame_counter = 0

        # ---- UI ----
        self.build_ui()
        self.update_frame()

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # ---- Video Area ----
        video_frame = ttk.LabelFrame(main_frame, text="Live Camera (AI overlay)")
        video_frame.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="nsew")
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ---- Right Panel ----
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)

        # Patient info
        patient_frame = ttk.LabelFrame(right_frame, text="Patient (Optional)")
        patient_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(patient_frame, text="Name:").grid(row=0, column=0, sticky="w")
        ttk.Label(patient_frame, text="Age:").grid(row=1, column=0, sticky="w")
        ttk.Label(patient_frame, text="ID:").grid(row=2, column=0, sticky="w")

        self.name_entry = ttk.Entry(patient_frame, width=20)
        self.age_entry = ttk.Entry(patient_frame, width=10)
        self.id_entry = ttk.Entry(patient_frame, width=20)

        self.name_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.age_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.id_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        patient_frame.columnconfigure(1, weight=1)

        # Result frame
        result_frame = ttk.LabelFrame(right_frame, text="AI Analysis (Live)")
        result_frame.grid(row=1, column=0, sticky="ew", pady=5)
        result_frame.columnconfigure(0, weight=1)

        self.pred_label_var = tk.StringVar(value="Risk: -")
        self.conf_label_var = tk.StringVar(value="Confidence: -")
        self.count_label_var = tk.StringVar(value="Detected eczema regions: -")

        self.pred_label = ttk.Label(
            result_frame,
            textvariable=self.pred_label_var,
            font=("Helvetica", 11, "bold"),
            wraplength=260
        )
        self.pred_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.conf_label = ttk.Label(result_frame, textvariable=self.conf_label_var)
        self.conf_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        self.count_label = ttk.Label(result_frame, textvariable=self.count_label_var)
        self.count_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)

        self.note_label = ttk.Label(
            result_frame,
            text="This tool does NOT provide a diagnosis.\n"
                 "Any suspicious finding should be evaluated by a dermatologist.",
            wraplength=260,
            foreground="red"
        )
        self.note_label.grid(row=3, column=0, sticky="w", padx=5, pady=4)

        # Buttons
        btn_frame = ttk.Frame(right_frame)
        btn_frame.grid(row=2, column=0, sticky="ew", pady=5)
        btn_frame.columnconfigure(0, weight=1)

        self.pdf_btn = ttk.Button(btn_frame, text="Generate PDF Report", command=self.generate_pdf_report)
        self.pdf_btn.grid(row=0, column=0, padx=5, pady=3, sticky="ew")

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_bgr = frame.copy()
                self.frame_counter += 1

                # Run YOLO every N frames for speed
                if self.model is not None and (self.frame_counter % INFERENCE_INTERVAL == 0):
                    has_eczema, best_conf, risk_label, annotated_bgr, box_count = analyze_with_yolo(
                        self.model,
                        self.current_frame_bgr
                    )

                    self.last_has_eczema = has_eczema
                    self.last_confidence = best_conf
                    self.last_risk_label = risk_label
                    self.last_box_count = box_count

                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    self.annotated_frame_pil = Image.fromarray(annotated_rgb)

                    # update right panel text
                    self.pred_label_var.set(f"Risk: {risk_label}")
                    if has_eczema and best_conf is not None and best_conf > 0:
                        self.conf_label_var.set(f"Confidence (top lesion): {best_conf * 100:.1f}%")
                    else:
                        self.conf_label_var.set("Confidence: -")
                    self.count_label_var.set(f"Detected eczema regions: {box_count if box_count else 0}")

                # --- Build live image with overlay text ---
                if self.annotated_frame_pil is not None:
                    live_img = np.array(self.annotated_frame_pil.convert("RGB"))
                else:
                    live_img = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2RGB)

                h, w, _ = live_img.shape

                # risk text for overlay
                if self.last_risk_label is not None:
                    risk_text = self.last_risk_label
                else:
                    risk_text = "No eczema detected"

                # white bar at top
                # ---- SEMI-TRANSPARENT RISK BAR ----
                bar_height = 32
                overlay = live_img.copy()
                alpha = 0.40  # 40% transparency

                # Draw solid white rectangle on overlay
                cv2.rectangle(overlay, (0, 0), (w, bar_height), (255, 255, 255), -1)

                # Blend with original frame
                cv2.addWeighted(overlay, alpha, live_img, 1 - alpha, 0, live_img)

                # ---- SMALLER RISK TEXT (20% smaller) ----
                cv2.putText(
                    live_img,
                    risk_text,
                    (10, int(bar_height * 0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,  # ↓ 20% smaller than 0.7
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                # show in Tk
                img_display = Image.fromarray(live_img)
                img_resized = img_display.resize((480, 360))
                imgtk = ImageTk.PhotoImage(image=img_resized)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    def _ensure_latest_analysis_for_report(self):
        """Make sure we have annotated frame + stats before PDF."""
        if self.model is None or self.current_frame_bgr is None:
            return
        if self.annotated_frame_pil is not None and self.last_risk_label is not None:
            return

        has_eczema, best_conf, risk_label, annotated_bgr, box_count = analyze_with_yolo(
            self.model,
            self.current_frame_bgr
        )
        self.last_has_eczema = has_eczema
        self.last_confidence = best_conf
        self.last_risk_label = risk_label
        self.last_box_count = box_count

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        self.annotated_frame_pil = Image.fromarray(annotated_rgb)

    def generate_pdf_report(self):
        if self.current_frame_bgr is None:
            messagebox.showinfo("No image", "No camera frame available to include in report.")
            return

        self._ensure_latest_analysis_for_report()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save Eczema Analysis Report"
        )
        if not file_path:
            return

        try:
            self._create_pdf(file_path)
            messagebox.showinfo("PDF created", f"Report saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("PDF error", f"Could not create PDF:\n{e}")

    def _create_pdf(self, file_path):
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4

        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 2 * cm, "PocketDoc – Eczema AI Assist Report")

        c.setFont("Helvetica", 10)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        c.drawString(2 * cm, height - 2.7 * cm, f"Date: {now_str}")

        # Patient info
        name = self.name_entry.get().strip()
        age = self.age_entry.get().strip()
        pid = self.id_entry.get().strip()

        y_pat = height - 3.5 * cm
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, y_pat, f"Patient Name: {name if name else '-'}")
        c.drawString(2 * cm, y_pat - 0.5 * cm, f"Age: {age if age else '-'}")
        c.drawString(2 * cm, y_pat - 1.0 * cm, f"Patient ID: {pid if pid else '-'}")

        # Save RAW + ANNOTATED
        tmp_dir = tempfile.gettempdir()
        raw_rgb = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2RGB)
        raw_pil = Image.fromarray(raw_rgb)
        raw_path = os.path.join(tmp_dir, "eczema_report_raw.jpg")
        raw_pil.save(raw_path, "JPEG")

        if self.annotated_frame_pil is not None:
            ann_pil = self.annotated_frame_pil
        else:
            ann_pil = raw_pil.copy()
        ann_path = os.path.join(tmp_dir, "eczema_report_annotated.jpg")
        ann_pil.save(ann_path, "JPEG")

        img_width = 7 * cm
        img_height = 7 * cm
        img_y = height - 14 * cm

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2 * cm, img_y + img_height + 0.4 * cm, "Unprocessed clinical photo")
        c.drawImage(raw_path, 2 * cm, img_y, width=img_width, height=img_height, preserveAspectRatio=True)

        c.drawString(2 * cm + img_width + 1.5 * cm,
                     img_y + img_height + 0.4 * cm,
                     "AI-processed (bounding boxes)")
        c.drawImage(
            ann_path,
            2 * cm + img_width + 1.5 * cm,
            img_y,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True
        )

        # AI result box
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, img_y - 0.7 * cm, "AI-Based Image Analysis (Not a Diagnosis)")

        c.setFont("Helvetica", 10)
        y_text = img_y - 1.3 * cm

        if self.last_risk_label is not None:
            c.drawString(2 * cm, y_text, f"Risk summary: {self.last_risk_label}")
            y_text -= 0.4 * cm
            if self.last_has_eczema and self.last_confidence is not None and self.last_confidence > 0:
                c.drawString(2 * cm, y_text, f"Top lesion confidence: {self.last_confidence * 100:.1f}%")
                y_text -= 0.4 * cm
            c.drawString(2 * cm, y_text, f"Detected eczema regions: {self.last_box_count if self.last_box_count else 0}")
            y_text -= 0.4 * cm
        else:
            c.drawString(2 * cm, y_text, "No AI analysis was performed on this frame.")
            y_text -= 0.4 * cm

        c.setFont("Helvetica-Oblique", 9)
        c.drawString(
            2 * cm,
            y_text,
            "Note: This AI-assisted report supports clinical judgment and does NOT replace a dermatology exam."
        )
        y_text -= 0.7 * cm

        # Eczema info & tips (your block)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y_text, "About Eczema (Dermatitis)")
        y_text -= 0.5 * cm

        c.setFont("Helvetica", 9)
        lines = [
            "Eczema is a chronic inflammatory skin condition that can cause redness, dryness,",
            "itching, cracking, and sometimes blisters or oozing lesions.",

            "",
            "Common eczema types:",
            "- Atopic dermatitis: Often starts in childhood; associated with asthma/allergies.",
            "- Contact dermatitis: Triggered by irritants or allergens (e.g., soaps, metals, cosmetics).",
            "- Nummular eczema: Coin-shaped itchy patches.",
            "- Seborrheic dermatitis: Flaky, oily areas (scalp, face, chest).",
            "- Dyshidrotic eczema: Small itchy blisters on hands/feet.",

            "",
            "Daily care & trigger management:",
            "- Use fragrance-free, gentle cleansers and thick moisturizers (creams/ointments).",
            "- Shower with lukewarm (not hot) water and limit time under water.",
            "- Apply moisturizer within 3–5 minutes after bathing.",
            "- Avoid scratching; keep nails short and consider cotton gloves at night.",
            "- Identify personal triggers (sweat, wool, harsh soaps, stress) and minimize exposure.",
            "- Prefer cotton clothing; avoid rough or synthetic fabrics.",

            "",
            "Red-flag symptoms – seek medical care promptly if:",
            "- Areas are very painful, rapidly worsening, or spreading.",
            "- You notice yellow crusts, pus, or fever (possible infection).",
            "- Eye involvement (red, painful, or blurred vision).",
            "- Eczema significantly affects sleep, school, work, or mood.",

            "",
            "Treatment options your dermatologist may consider:",
            "- Topical anti-inflammatory creams (steroids or non-steroid creams).",
            "- Short courses of oral medications for flares when appropriate.",
            "- Antihistamines for itch relief in select cases.",
            "- Phototherapy or systemic treatments for severe, chronic disease.",

            "",
            "Reminder:",
            "This report is generated automatically from a single image and is for informational use only.",
            "It is not a diagnosis or prescription. Please consult a dermatologist or healthcare provider",
            "for personal evaluation and treatment planning."
        ]

        max_chars = 95
        for line in lines:
            if len(line) <= max_chars:
                to_draw = [line]
            else:
                to_draw = []
                words = line.split()
                current = ""
                for w in words:
                    if len(current) + len(w) + 1 <= max_chars:
                        current = (current + " " + w).strip()
                    else:
                        to_draw.append(current)
                        current = w
                if current:
                    to_draw.append(current)

            for l in to_draw:
                if y_text < 2 * cm:
                    c.showPage()
                    y_text = height - 2 * cm
                    c.setFont("Helvetica", 9)
                c.drawString(2 * cm, y_text, l)
                y_text -= 0.4 * cm

        c.showPage()
        c.save()

    def on_close(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EczemaApp(root)
    root.mainloop()
