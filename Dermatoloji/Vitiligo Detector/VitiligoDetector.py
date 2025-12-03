import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import tempfile
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------- CONFIG ----------------

MODEL_PATH = r"/Users/avnitan/PycharmProjects/TestProject/PocketDoc/Modeller/Dermatology/Vitiligo/vitiligo.pt"

# Brand-ish colors
COLOR_BG_MAIN = "#e6f2ff"
COLOR_PRIMARY = "#003366"
COLOR_ACCENT = "#ffffff"


# ---------------- HELPERS ----------------

def safe_register_font():
    """
    Try to register League Spartan; fall back silently if not found.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(
            script_dir,
            "..",
            "..",
            "Fonts",
            "League_Spartan",
            "static",
            "LeagueSpartan-SemiBold.ttf",
        )
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("LeagueSpartan", font_path))
            return "LeagueSpartan"
    except Exception:
        pass
    return "Helvetica"


def summarize_vitiligo(area_ratio, mean_conf, n_boxes):
    """
    Build a human-readable summary string and severity label.
    area_ratio: 0-100 (%)
    """
    if n_boxes == 0 or area_ratio < 0.3:
        label = "No obvious vitiligo area detected"
        detail = (
            "No clear vitiligo-like depigmented patches are detected on this frame. "
            "Lighting, camera quality, and skin type can affect detection."
        )
    elif area_ratio < 5:
        label = "Localized vitiligo-like areas"
        detail = (
            "Small, localized depigmented patches are detected. "
            "These may correspond to early or limited vitiligo, "
            "but only a dermatologist can confirm this."
        )
    elif area_ratio < 15:
        label = "Mild–moderate vitiligo-like involvement"
        detail = (
            "Multiple depigmented patches are detected, suggesting mild–moderate involvement. "
            "Regular follow-up with a dermatologist is recommended."
        )
    else:
        label = "Extended vitiligo-like involvement"
        detail = (
            "A relatively larger proportion of skin shows vitiligo-like depigmentation. "
            "You should consult a dermatologist for detailed evaluation and treatment planning."
        )

    conf_text = (
        f"Average model confidence: {mean_conf:.1f}%\n"
        if n_boxes > 0
        else "Average model confidence: —\n"
    )

    extra = (
        "- This tool is for educational screening only and does NOT provide a diagnosis.\n"
        "- Diagnosis and treatment decisions must always be made by a dermatologist.\n"
        "- Try capturing images under uniform lighting, without strong shadows."
    )

    full = (
        f"{label}\n\n"
        f"Detected vitiligo-like area: {area_ratio:.1f}% of visible skin region (approx.)\n"
        f"{conf_text}"
        f"Number of vitiligo-like regions: {n_boxes}\n\n"
        f"{detail}\n\n"
        f"{extra}"
    )

    return label, full


# ---------------- MAIN APP ----------------

class VitiligoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PocketDoc – Vitiligo Detector (Prototype)")
        self.configure(bg=COLOR_BG_MAIN)

        self.cap = None
        self.model = None

        self.current_frame = None
        self.annotated_frame = None
        self.last_area_ratio = 0.0
        self.last_mean_conf = 0.0
        self.last_n_boxes = 0

        self._load_model()
        self._build_ui()
        self._open_camera()
        self._update_loop()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- Setup ----------

    def _load_model(self):
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load vitiligo model:\n{e}")
            self.model = None

    def _open_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error", "Could not open webcam. Please check your camera."
            )
            self.cap = None

    # ---------- UI ----------

    def _build_ui(self):
        # Main frames
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        self.style = ttk.Style()
        self.style.configure(
            "TFrame", background=COLOR_BG_MAIN
        )
        self.style.configure(
            "TLabel", background=COLOR_BG_MAIN, foreground=COLOR_PRIMARY, font=("Helvetica", 11)
        )
        self.style.configure(
            "Header.TLabel",
            background=COLOR_BG_MAIN,
            foreground=COLOR_PRIMARY,
            font=("Helvetica", 18, "bold"),
        )
        self.style.configure(
            "TButton",
            font=("Helvetica", 11, "bold"),
        )

        # Left: Video
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        title_lbl = ttk.Label(
            left_frame, text="Live Vitiligo Detection", style="Header.TLabel"
        )
        title_lbl.pack(pady=(0, 10))

        self.video_label = tk.Label(left_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # Right: Controls & info
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # Patient info
        info_frame = ttk.LabelFrame(right_frame, text="Patient Information")
        info_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(info_frame, text="Name / ID:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Label(info_frame, text="Age:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        ttk.Label(info_frame, text="Notes:").grid(row=2, column=0, sticky="nw", padx=5, pady=3)

        self.entry_name = ttk.Entry(info_frame)
        self.entry_age = ttk.Entry(info_frame)
        self.text_notes = tk.Text(info_frame, height=3, width=28, wrap="word")

        self.entry_name.grid(row=0, column=1, sticky="ew", padx=5, pady=3)
        self.entry_age.grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        self.text_notes.grid(row=2, column=1, sticky="ew", padx=5, pady=3)

        info_frame.columnconfigure(1, weight=1)

        # Summary
        summary_frame = ttk.LabelFrame(right_frame, text="Model Summary")
        summary_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.label_area = ttk.Label(summary_frame, text="Vitiligo-like area: 0.0%")
        self.label_conf = ttk.Label(summary_frame, text="Average confidence: —")
        self.label_regions = ttk.Label(summary_frame, text="Regions detected: 0")

        self.label_area.pack(anchor="w", padx=5, pady=2)
        self.label_conf.pack(anchor="w", padx=5, pady=2)
        self.label_regions.pack(anchor="w", padx=5, pady=2)

        ttk.Label(summary_frame, text="Detailed explanation:").pack(
            anchor="w", padx=5, pady=(8, 2)
        )
        self.text_summary = tk.Text(
            summary_frame, height=10, wrap="word", bg=COLOR_ACCENT
        )
        self.text_summary.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        self.text_summary.insert(
            "1.0",
            "Live vitiligo-like area analysis will appear here.\n\n"
            "Position the camera so that the affected skin area is clearly visible "
            "and well-lit. Then capture a moment you want to report and generate a PDF.",
        )
        self.text_summary.config(state="disabled")

        # Buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill="x", pady=(0, 5))

        self.btn_capture = ttk.Button(
            button_frame, text="Create PDF Report (Current Frame)", command=self.generate_pdf
        )
        self.btn_capture.pack(side="left", padx=(0, 5))

        self.btn_quit = ttk.Button(
            button_frame, text="Quit", command=self.on_close
        )
        self.btn_quit.pack(side="right", padx=(5, 0))

    # ---------- Video + Detection ----------

    def _update_loop(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self._run_detection_and_draw(frame)
                # Show in Tkinter
                self._show_frame(self.annotated_frame if self.annotated_frame is not None else frame)

        self.after(20, self._update_loop)

    def _run_detection_and_draw(self, frame):
        if self.model is None:
            self.annotated_frame = frame
            return

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        try:
            results = self.model(frame, verbose=False)[0]
        except Exception as e:
            print(f"Model inference error: {e}")
            self.annotated_frame = frame
            return

        boxes = results.boxes
        vit_area = 0.0
        confs = []
        n_boxes = 0

        annotated = frame.copy()

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf[0].cpu().numpy())
                n_boxes += 1
                confs.append(conf)

                box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                vit_area += box_area

                # Draw bounding box
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                label = f"Vitiligo {conf*100:.1f}%"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1) - th - baseline),
                    (int(x1) + tw, int(y1)),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (int(x1), int(y1) - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        area_ratio = (vit_area / frame_area) * 100.0 if frame_area > 0 else 0.0
        mean_conf = (np.mean(confs) * 100.0) if confs else 0.0

        self.last_area_ratio = area_ratio
        self.last_mean_conf = mean_conf
        self.last_n_boxes = n_boxes

        self.annotated_frame = annotated

        # Update summary labels & text
        self.label_area.config(text=f"Vitiligo-like area: {area_ratio:.1f}%")
        self.label_conf.config(
            text="Average confidence: —" if n_boxes == 0 else f"Average confidence: {mean_conf:.1f}%"
        )
        self.label_regions.config(text=f"Regions detected: {n_boxes}")

        title, text = summarize_vitiligo(area_ratio, mean_conf, n_boxes)
        self.text_summary.config(state="normal")
        self.text_summary.delete("1.0", "end")
        self.text_summary.insert("1.0", text)
        self.text_summary.config(state="disabled")

    def _show_frame(self, frame):
        # Resize to fit UI nicely (keep aspect)
        max_w, max_h = 800, 600
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # ---------- PDF Report ----------

    def generate_pdf(self):
        if self.annotated_frame is None:
            messagebox.showwarning(
                "No Frame",
                "No frame available yet. Please wait for the camera to start and try again.",
            )
            return

        # Ask where to save
        default_name = f"VitiligoReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=default_name,
            title="Save Vitiligo Report as...",
        )
        if not filepath:
            return

        # Save annotated image to temp
        tmp_dir = tempfile.mkdtemp(prefix="vitiligo_report_")
        img_path = os.path.join(tmp_dir, "annotated_frame.png")
        cv2.imwrite(img_path, self.annotated_frame)

        font_name = safe_register_font()

        c = canvas.Canvas(filepath, pagesize=A4)
        w, h = A4
        margin = 40

        # Header
        c.setFont(font_name, 18)
        c.setFillColorRGB(0, 0.2, 0.4)
        c.drawString(margin, h - margin - 10, "PocketDoc – Vitiligo Screening Report")

        c.setFont(font_name, 10)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(
            margin,
            h - margin - 30,
            f"Date / Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        # Patient info
        patient_name = self.entry_name.get().strip()
        age = self.entry_age.get().strip()
        notes = self.text_notes.get("1.0", "end").strip()

        y = h - margin - 60
        c.setFont(font_name, 11)
        c.drawString(margin, y, f"Patient: {patient_name or '-'}")
        y -= 15
        c.drawString(margin, y, f"Age: {age or '-'}")
        y -= 15
        c.drawString(margin, y, f"Notes: {notes or '-'}")
        y -= 25

        # Image
        img_width = w - 2 * margin
        img_height = img_width * 0.6  # keep reasonable aspect on page
        c.drawImage(
            img_path,
            margin,
            y - img_height,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        y = y - img_height - 20

        # Model summary
        title, text_summary = summarize_vitiligo(
            self.last_area_ratio, self.last_mean_conf, self.last_n_boxes
        )

        c.setFont(font_name, 12)
        c.setFillColorRGB(0, 0.2, 0.4)
        c.drawString(margin, y, "Automated Vitiligo-like Area Analysis")
        y -= 18

        c.setFont(font_name, 10)
        c.setFillColorRGB(0, 0, 0)

        summary_lines = [
            f"Detected vitiligo-like area: {self.last_area_ratio:.1f}% of visible region (approx.)",
            f"Average model confidence: {self.last_mean_conf:.1f}%" if self.last_n_boxes > 0 else
            "Average model confidence: —",
            f"Regions detected: {self.last_n_boxes}",
            "",
        ]
        for line in summary_lines:
            c.drawString(margin, y, line)
            y -= 13

        # Vitiligo info block
        y -= 5
        c.setFont(font_name, 11)
        c.setFillColorRGB(0, 0.2, 0.4)
        c.drawString(margin, y, "About Vitiligo")
        y -= 15

        c.setFont(font_name, 9)
        c.setFillColorRGB(0, 0, 0)
        info_text = [
            "Vitiligo is a chronic skin condition characterized by loss of pigment (melanin), "
            "leading to well-defined white patches on the skin.",
            "It is thought to be autoimmune in origin and may be associated with other autoimmune diseases.",
            "Common treatment options include topical therapies, phototherapy, and, in selected cases, "
            "surgical or laser-based procedures.",
            "",
            "This AI tool provides an approximate geometric estimation of depigmented areas "
            "on the captured image. It does NOT establish a diagnosis and does NOT replace "
            "clinical examination.",
            "",
            "If you notice spreading white patches, color changes, or have cosmetic concerns, "
            "please consult a dermatologist for a full evaluation and personalized treatment plan.",
        ]

        for paragraph in info_text:
            # Simple word-wrapped text drawing
            words = paragraph.split(" ")
            line = ""
            for word in words:
                test_line = (line + " " + word).strip()
                if c.stringWidth(test_line, font_name, 9) > (w - 2 * margin):
                    c.drawString(margin, y, line)
                    y -= 11
                    line = word
                else:
                    line = test_line
            if line:
                c.drawString(margin, y, line)
                y -= 11
            y -= 3

        y -= 5
        c.setFont(font_name, 9)
        c.setFillColorRGB(0.6, 0, 0)
        disclaimers = [
            "Disclaimer:",
            "- This report is generated automatically by an AI-based image analysis tool.",
            "- It is intended only for educational / research use and does not constitute medical advice.",
            "- Diagnosis and treatment decisions must always be made by a qualified dermatologist.",
        ]
        for line in disclaimers:
            c.drawString(margin, y, line)
            y -= 11

        c.showPage()
        c.save()

        messagebox.showinfo(
            "Report Saved",
            f"Vitiligo screening report has been saved as:\n{filepath}",
        )

    # ---------- Cleanup ----------

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()


if __name__ == "__main__":
    app = VitiligoApp()
    app.mainloop()
