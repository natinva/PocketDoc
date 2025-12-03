import os
import cv2
import numpy as np
from datetime import datetime
import tempfile

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from ultralytics import YOLO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------- CONFIG ----------------

MODEL_PATH = "/Users/avnitan/PycharmProjects/TestProject/PocketDoc/Modeller/Orthopaedics and Traumatology/ACL Detection Sagittal MRI/acl-detector.pt"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEAGUE_SPARTAN_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "..",
    "Fonts",
    "League_Spartan",
    "static",
    "LeagueSpartan-SemiBold.ttf"
)

PDF_FONT_NAME = "Helvetica"
if os.path.exists(LEAGUE_SPARTAN_PATH):
    try:
        pdfmetrics.registerFont(TTFont("LeagueSpartan", LEAGUE_SPARTAN_PATH))
        PDF_FONT_NAME = "LeagueSpartan"
    except Exception:
        PDF_FONT_NAME = "Helvetica"

CLASS_MAP = {
    0: "Complete ACL rupture",
    1: "Intact / normal ACL",
    2: "Partial ACL injury",
}

CLASS_COLORS = {
    0: (0, 0, 255),    # red
    1: (0, 200, 0),    # green
    2: (0, 215, 255),  # yellow
}

# ---------------- EXPLANATION TEXTS ----------------

GENERAL_INFO = [
    "The anterior cruciate ligament (ACL) is a key stabilizer of the knee,",
    "resisting anterior translation and rotational instability of the tibia.",
    "",
    "Main stabilizers of the knee:",
    "- Static stabilizers: ACL, PCL, MCL, LCL, joint capsule, menisci.",
    "- Dynamic stabilizers: quadriceps, hamstrings, gastrocnemius, and hip muscles.",
    "",
    "ACL injury mechanisms:",
    "- Non-contact pivoting, sudden deceleration, or landing from a jump.",
    "- Direct contact trauma to the knee.",
]

CLASS_EXPLANATIONS = {
    0: [
        "Interpretation:",
        "- Model suggests a complete ACL rupture in the detected region.",
        "",
        "Clinical considerations:",
        "- Often associated with knee giving way, swelling, and instability.",
        "- Risk of secondary meniscal and cartilage injuries if left untreated.",
        "",
        "Suggested next steps:",
        "- Orthopedic evaluation is strongly recommended.",
        "- Clinical exam (Lachman, pivot-shift) and correlation with all MRI slices.",
        "- Discuss non-operative vs. surgical reconstruction based on age, activity level,",
        "  associated injuries, and patient expectations.",
    ],
    1: [
        "Interpretation:",
        "- Model suggests the ACL appears intact in the detected region.",
        "",
        "Clinical considerations:",
        "- Normal-appearing ACL does not fully exclude other pathologies (e.g., meniscal,",
        "  cartilage, or collateral ligament injuries).",
        "",
        "Suggested next steps:",
        "- If symptoms persist, a full orthopedic assessment is advised.",
        "- Clinical examination and review of the entire MRI series remain essential.",
    ],
    2: [
        "Interpretation:",
        "- Model suggests a partial ACL injury in the detected region.",
        "",
        "Clinical considerations:",
        "- Partial tears can still cause instability, pain, or reduced sports performance.",
        "- Risk of progression to complete rupture in high-demand sports.",
        "",
        "Suggested next steps:",
        "- Orthopedic consultation is recommended.",
        "- Consider structured physiotherapy focusing on quadriceps/hamstrings strength and",
        "  neuromuscular control.",
        "- Management (conservative vs. surgical) should be individualized.",
    ],
}

DISCLAIMER_LINES = [
    "Important:",
    "- This tool provides AI-assisted image analysis and does NOT replace a full clinical,",
    "  radiological, and orthopedic evaluation.",
    "- All decisions must be made by a qualified healthcare professional.",
]

# ---------------- HELPER FUNCTIONS ----------------


def cv2_to_tkimage(bgr_img, max_size=(640, 640)):
    if bgr_img is None:
        return None
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.thumbnail(max_size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


def draw_boxes(image_bgr, results, min_conf=0.20):
    """
    Draw a single box.
    Priority:
      1) any class 0 or 2 (injured) with conf >= min_conf
      2) otherwise highest-conf class 1 (intact) with conf >= min_conf
      3) if nothing >= min_conf -> no detection
    """
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return image_bgr.copy(), None

    boxes = results.boxes
    confs = boxes.conf.cpu().numpy()
    clses = boxes.cls.cpu().numpy().astype(int)

    valid = confs >= min_conf
    if not valid.any():
        return image_bgr.copy(), None

    # Prefer injured (0,2) if present
    inj_mask = np.isin(clses, [0, 2]) & valid
    if inj_mask.any():
        idx = int(np.argmax(confs * inj_mask))
    else:
        # no injured, fall back to highest-conf valid (likely intact)
        idx = int(np.argmax(confs * valid))

    b = boxes[idx]
    x1, y1, x2, y2 = b.xyxy.cpu().numpy().astype(int)[0]
    cls_id = int(b.cls.item())
    conf = float(b.conf.item())

    color = CLASS_COLORS.get(cls_id, (0, 255, 0))
    label = CLASS_MAP.get(cls_id, f"Class {cls_id}")
    label_text = f"{label} ({conf * 100:.1f}%)"

    out = image_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        out,
        label_text,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    detection_info = {
        "cls_id": cls_id,
        "class_name": label,
        "confidence": conf,
        "bbox": (x1, y1, x2, y2),
    }

    return out, detection_info


def build_text_summary(detection_info, image_label):
    lines = []

    lines.append("ACL Sagittal MRI Analysis")
    lines.append("-" * 35)
    lines.append(f"Source: {image_label}")
    lines.append(f"Report time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if detection_info is None:
        lines.append("")
        lines.append("No clear ACL region was automatically detected on this frame.")
        lines.append("Please verify slice selection and correlate with full MRI and clinical exam.")
        lines.append("")
    else:
        cls_id = detection_info["cls_id"]
        cls_name = detection_info["class_name"]
        conf = detection_info["confidence"]

        lines.append("")
        lines.append(f"AI prediction: {cls_name}")
        lines.append(f"Confidence: {conf * 100:.1f}%")
        lines.append("")
        lines.extend(GENERAL_INFO)
        lines.append("")
        lines.extend(CLASS_EXPLANATIONS.get(cls_id, []))

    lines.append("")
    lines.extend(DISCLAIMER_LINES)

    return "\n".join(lines)


def save_temp_png_from_bgr(bgr_img):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)
    return path


def create_pdf_report(
    save_path,
    original_bgr,
    processed_bgr,
    patient_info_text,
    summary_text,
):
    c = pdf_canvas.Canvas(save_path, pagesize=A4)
    width, height = A4

    margin_x = 40
    margin_y = 40

    # Title
    c.setFont(PDF_FONT_NAME, 18)
    c.drawString(margin_x, height - margin_y, "ACL Sagittal MRI AI Report")

    c.setFont(PDF_FONT_NAME, 10)
    c.drawRightString(
        width - margin_x,
        height - margin_y + 2,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    y = height - margin_y - 30

    # Patient info
    c.setFont(PDF_FONT_NAME, 11)
    for line in patient_info_text.split("\n"):
        c.drawString(margin_x, y, line)
        y -= 14
    y -= 10

    # Images
    img_height = 220
    img_width = (width - 3 * margin_x) / 2

    orig_path = proc_path = None

    if original_bgr is not None:
        orig_path = save_temp_png_from_bgr(original_bgr)
        c.drawImage(
            orig_path,
            margin_x,
            y - img_height,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True,
            anchor="sw",
        )

    if processed_bgr is not None:
        proc_path = save_temp_png_from_bgr(processed_bgr)
        c.drawImage(
            margin_x + img_width + margin_x,
            y - img_height,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True,
            anchor="sw",
        )

    c.setFont(PDF_FONT_NAME, 11)
    c.drawString(margin_x, y - img_height - 14, "Original sagittal MRI frame")
    c.drawString(
        margin_x + img_width + margin_x,
        y - img_height - 14,
        "AI-annotated MRI frame",
    )

    y = y - img_height - 40

    # Summary text
    c.setFont(PDF_FONT_NAME, 11)
    text_obj = c.beginText()
    text_obj.setTextOrigin(margin_x, y)
    text_obj.setLeading(14)

    for line in summary_text.split("\n"):
        if not line.strip():
            text_obj.textLine("")
        else:
            text_obj.textLine(line[:1000])
    c.drawText(text_obj)

    for p in [orig_path, proc_path]:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    c.showPage()
    c.save()


# ---------------- MAIN APP ----------------


class ACLDetectorLiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PocketDoc - ACL MRI Live Detector")

        # Model
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model:\n{e}")
            raise

        # Camera
        self.cap = None
        self.cam_running = False

        # Last analyzed frame
        self.last_source = "none"  # "camera" or "image"
        self.original_bgr = None
        self.processed_bgr = None
        self.detection_info = None
        self.current_image_label = "N/A"

        # Tk image ref
        self.tk_video = None

        self.build_ui()
        self.start_camera()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        self.root.geometry("1150x650")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: single processed live view
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(left_frame, text="Live / Processed View").pack(anchor="w", padx=5, pady=5)
        self.canvas_video = ttk.Label(left_frame)
        self.canvas_video.pack(fill="both", expand=True, padx=5, pady=5)

        # Right side
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=False)

        # Patient info
        patient_frame = ttk.LabelFrame(right_frame, text="Patient Info")
        patient_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(patient_frame, text="Name:").grid(row=0, column=0, sticky="e")
        ttk.Label(patient_frame, text="Age:").grid(row=1, column=0, sticky="e")
        ttk.Label(patient_frame, text="Sex:").grid(row=2, column=0, sticky="e")
        ttk.Label(patient_frame, text="ID:").grid(row=3, column=0, sticky="e")

        self.entry_name = ttk.Entry(patient_frame, width=22)
        self.entry_age = ttk.Entry(patient_frame, width=22)
        self.entry_sex = ttk.Entry(patient_frame, width=22)
        self.entry_id = ttk.Entry(patient_frame, width=22)

        self.entry_name.grid(row=0, column=1, padx=3, pady=2)
        self.entry_age.grid(row=1, column=1, padx=3, pady=2)
        self.entry_sex.grid(row=2, column=1, padx=3, pady=2)
        self.entry_id.grid(row=3, column=1, padx=3, pady=2)

        # Summary
        summary_frame = ttk.LabelFrame(right_frame, text="AI Analysis Summary")
        summary_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.text_summary = tk.Text(summary_frame, width=42, wrap="word")
        self.text_summary.pack(fill="both", expand=True)
        self.text_summary.insert(
            "1.0",
            "Live ACL detector is running.\n\n"
            "The last processed frame (camera or uploaded image) will be used\n"
            "when you create a PDF report.\n\n"
            "Reminder: Always correlate with full MRI series and clinical exam.",
        )
        self.text_summary.config(state="disabled")

        # Buttons
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)

        self.btn_start_cam = ttk.Button(btn_frame, text="Start Camera", command=self.start_camera)
        self.btn_stop_cam = ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera)
        self.btn_load_img = ttk.Button(btn_frame, text="Upload MRI Image", command=self.load_image)
        self.btn_report = ttk.Button(btn_frame, text="Create PDF Report", command=self.create_report)
        self.btn_quit = ttk.Button(btn_frame, text="Quit", command=self.on_close)

        self.btn_start_cam.grid(row=0, column=0, padx=3, pady=3, sticky="we")
        self.btn_stop_cam.grid(row=0, column=1, padx=3, pady=3, sticky="we")
        self.btn_load_img.grid(row=1, column=0, padx=3, pady=3, columnspan=2, sticky="we")
        self.btn_report.grid(row=2, column=0, padx=3, pady=3, columnspan=2, sticky="we")
        self.btn_quit.grid(row=3, column=0, padx=3, pady=3, columnspan=2, sticky="we")

    # ------------- Camera loop -------------

    def start_camera(self):
        if self.cam_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            self.cap = None
            return

        self.cam_running = True
        self.update_camera_frame()

    def stop_camera(self):
        self.cam_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_camera_frame(self):
        if not self.cam_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        img = frame.copy()

        try:
            # lowered to 0.20 as requested
            results = self.model(img, imgsz=640, conf=0.20)[0]
        except Exception as e:
            self.stop_camera()
            messagebox.showerror("Inference Error", f"Problem during model inference:\n{e}")
            return

        processed, det_info = draw_boxes(img, results, min_conf=0.20)

        # store last analyzed frame
        self.last_source = "camera"
        self.original_bgr = img
        self.processed_bgr = processed
        self.detection_info = det_info
        self.current_image_label = "Live camera frame"

        # show processed only
        self.tk_video = cv2_to_tkimage(self.processed_bgr)
        self.canvas_video.configure(image=self.tk_video)

        # summary
        summary = build_text_summary(self.detection_info, self.current_image_label)
        self.text_summary.config(state="normal")
        self.text_summary.delete("1.0", "end")
        self.text_summary.insert("1.0", summary)
        self.text_summary.config(state="disabled")

        self.root.after(50, self.update_camera_frame)  # ~20 FPS

    # ------------- Image upload -------------

    def load_image(self):
        # macOS-safe: space-separated patterns, no semicolons
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Select sagittal MRI slice",
            filetypes=filetypes
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image. Please select a standard image file.")
            return

        # optional: stop camera while viewing uploaded MRI
        self.stop_camera()

        try:
            results = self.model(img, imgsz=640, conf=0.20)[0]
        except Exception as e:
            messagebox.showerror("Inference Error", f"Problem during model inference:\n{e}")
            return

        processed, det_info = draw_boxes(img, results, min_conf=0.20)

        self.last_source = "image"
        self.original_bgr = img
        self.processed_bgr = processed
        self.detection_info = det_info
        self.current_image_label = f"Uploaded: {os.path.basename(path)}"

        self.tk_video = cv2_to_tkimage(self.processed_bgr)
        self.canvas_video.configure(image=self.tk_video)

        summary = build_text_summary(self.detection_info, self.current_image_label)
        self.text_summary.config(state="normal")
        self.text_summary.delete("1.0", "end")
        self.text_summary.insert("1.0", summary)
        self.text_summary.config(state="disabled")

    # ------------- Report -------------

    def get_patient_info_text(self):
        name = self.entry_name.get().strip()
        age = self.entry_age.get().strip()
        sex = self.entry_sex.get().strip()
        pid = self.entry_id.get().strip()

        lines = ["Patient information:"]
        if name:
            lines.append(f"- Name: {name}")
        if age:
            lines.append(f"- Age: {age}")
        if sex:
            lines.append(f"- Sex: {sex}")
        if pid:
            lines.append(f"- ID: {pid}")
        if not (name or age or sex or pid):
            lines.append("- Not provided")

        return "\n".join(lines)

    def create_report(self):
        if self.original_bgr is None or self.processed_bgr is None:
            messagebox.showwarning(
                "No Frame",
                "No analyzed frame found.\n\n"
                "Use live camera or upload an MRI image first."
            )
            return

        patient_info = self.get_patient_info_text()
        summary = build_text_summary(self.detection_info, self.current_image_label)

        save_path = filedialog.asksaveasfilename(
            title="Save PDF report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="ACL_MRI_Report.pdf",
        )
        if not save_path:
            return

        try:
            create_pdf_report(
                save_path,
                self.original_bgr,
                self.processed_bgr,
                patient_info,
                summary,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not create PDF report:\n{e}")
            return

        messagebox.showinfo("Report created", f"PDF report saved to:\n{save_path}")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ACLDetectorLiveApp(root)
    root.mainloop()
