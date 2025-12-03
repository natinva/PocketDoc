import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import mediapipe as mp
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image, ImageTk
import tempfile
import os


# ---------- GEOMETRY HELPERS ----------

def angle_between(v1, v2):
    """Return angle in degrees between two vectors."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def carrying_angle_from_landmarks(landmarks, image_shape, side="right"):
    """
    Approximate carrying angle from MediaPipe Pose landmarks.
    carrying_angle ≈ 180° - angle(humerus vs forearm)
    """
    h, w, _ = image_shape
    mp_pose = mp.solutions.pose

    if side.lower() == "right":
        idx_sh = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        idx_el = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        idx_wr = mp_pose.PoseLandmark.RIGHT_WRIST.value
    else:
        idx_sh = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        idx_el = mp_pose.PoseLandmark.LEFT_ELBOW.value
        idx_wr = mp_pose.PoseLandmark.LEFT_WRIST.value

    try:
        p_sh = landmarks[idx_sh]
        p_el = landmarks[idx_el]
        p_wr = landmarks[idx_wr]
    except IndexError:
        return None

    shoulder = np.array([p_sh.x * w, p_sh.y * h])
    elbow = np.array([p_el.x * w, p_el.y * h])
    wrist = np.array([p_wr.x * w, p_wr.y * h])

    upper = shoulder - elbow
    forearm = wrist - elbow

    elbow_angle = angle_between(upper, forearm)
    if elbow_angle is None:
        return None

    carrying_angle = 180.0 - elbow_angle
    return carrying_angle


# ---------- PDF REPORT ----------

def create_pdf_report(
    healthy_angle,
    injured_angle,
    loss_angle,
    healthy_raw,
    healthy_proc,
    injured_raw,
    injured_proc,
    filename,
    patient_info=None
):
    # Save images to temporary files
    temp_files = []

    def save_temp_image(img, prefix):
        fd, path = tempfile.mkstemp(suffix=".png", prefix=prefix)
        os.close(fd)
        cv2.imwrite(path, img)
        temp_files.append(path)
        return path

    h_raw_path = save_temp_image(healthy_raw, "healthy_raw_")
    h_proc_path = save_temp_image(healthy_proc, "healthy_proc_")
    i_raw_path = save_temp_image(injured_raw, "injured_raw_")
    i_proc_path = save_temp_image(injured_proc, "injured_proc_")

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    margin = 40
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "Elbow Carrying Angle Report")
    y -= 25

    # Patient info
    c.setFont("Helvetica", 11)
    if patient_info:
        p_lines = [
            f"Patient: {patient_info.get('name', '')}",
            f"Age: {patient_info.get('age', '')}   Sex: {patient_info.get('sex', '')}",
            f"Patient ID: {patient_info.get('id', '')}",
            ""
        ]
        for line in p_lines:
            c.drawString(margin, y, line)
            y -= 14

    # Summary text
    text_lines = [
        "This report summarizes an automated measurement of the elbow carrying angle.",
        "",
        "The carrying angle is the physiologic valgus angle between the arm and forearm in full extension.",
        "In most individuals it is typically around 5–15 degrees, and it can differ slightly between sides.",
        "",
        f"Measured healthy side angle: {healthy_angle:.1f} degrees",
        f"Measured injured side angle: {injured_angle:.1f} degrees",
        f"Carrying angle loss (side-to-side difference): {loss_angle:.1f} degrees",
        "",
        "Interpretation:",
        "- Small differences are common and may be clinically insignificant.",
        "- Larger losses of carrying angle can be associated with malalignment after elbow injury or fracture.",
        "",
        "NOTE: This tool is for documentation and educational purposes and does not replace a formal",
        "orthopedic clinical examination or radiographic assessment."
    ]

    for line in text_lines:
        c.drawString(margin, y, line)
        y -= 14
        if y < 200:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 11)

    # New section for images
    if y < 320:
        c.showPage()
        y = height - margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Captured Images")
    y -= 20

    img_width = (width - 3 * margin) / 2
    img_height = img_width * 0.75  # approximate aspect ratio

    # Healthy images
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, "Healthy side (raw):")
    c.drawImage(
        h_raw_path,
        margin,
        y - img_height - 5,
        width=img_width,
        height=img_height,
        preserveAspectRatio=True,
        mask='auto'
    )

    c.drawString(margin + img_width + margin, y, "Healthy side (processed):")
    c.drawImage(
        h_proc_path,
        margin + img_width + margin,
        y - img_height - 5,
        width=img_width,
        height=img_height,
        preserveAspectRatio=True,
        mask='auto'
    )

    y = y - img_height - 40

    if y < 200:
        c.showPage()
        y = height - margin

    # Injured images
    c.drawString(margin, y, "Injured side (raw):")
    c.drawImage(
        i_raw_path,
        margin,
        y - img_height - 5,
        width=img_width,
        height=img_height,
        preserveAspectRatio=True,
        mask='auto'
    )

    c.drawString(margin + img_width + margin, y, "Injured side (processed):")
    c.drawImage(
        i_proc_path,
        margin + img_width + margin,
        y - img_height - 5,
        width=img_width,
        height=img_height,
        preserveAspectRatio=True,
        mask='auto'
    )

    c.showPage()
    c.save()

    # Clean up temp files
    for path in temp_files:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------- MAIN TKINTER APP ----------

class CarryingAngleApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Elbow Carrying Angle Measurement")
        self.configure(bg="#e6f2ff")
        # a bit larger to accommodate 20% bigger camera
        self.geometry("1300x800")
        self.minsize(1200, 750)

        # Video + pose
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Current frame info
        self.current_raw_frame = None
        self.current_processed_frame = None
        self.current_right_angle = None
        self.current_left_angle = None

        # Captured data
        self.healthy_angle = None
        self.injured_angle = None
        self.healthy_raw = None
        self.healthy_proc = None
        self.injured_raw = None
        self.injured_proc = None

        self._configure_style()
        self._create_widgets()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start updating camera
        self.update_frame()

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Main.TFrame", background="#e6f2ff")
        style.configure("Bottom.TFrame", background="#e6f2ff")
        style.configure("Card.TFrame", background="white", relief="groove", borderwidth=1)

        style.configure(
            "Title.TLabel",
            background="#e6f2ff",
            foreground="#003366",
            font=("Helvetica", 18, "bold")
        )
        style.configure(
            "Subtitle.TLabel",
            background="#e6f2ff",
            foreground="#003366",
            font=("Helvetica", 11)
        )
        style.configure(
            "CardTitle.TLabel",
            background="white",
            foreground="#003366",
            font=("Helvetica", 12, "bold")
        )
        style.configure(
            "CardText.TLabel",
            background="white",
            foreground="#003366",
            font=("Helvetica", 10)
        )
        style.configure(
            "TLabel",
            background="#e6f2ff",
            foreground="#003366",
            font=("Helvetica", 10)
        )

        style.configure(
            "Primary.TButton",
            font=("Helvetica", 11, "bold"),
            padding=6
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#004b99"), ("!disabled", "#003366")],
            foreground=[("!disabled", "white")]
        )

        style.configure(
            "Secondary.TButton",
            font=("Helvetica", 10),
            padding=4
        )

    def _create_widgets(self):
        # Main vertical layout: video on top, bottom controls bar
        main_frame = ttk.Frame(self, style="Main.TFrame")
        main_frame.pack(fill="both", expand=True)

        # Header above video
        header = ttk.Frame(main_frame, style="Main.TFrame")
        header.pack(fill="x", pady=(10, 0))

        ttk.Label(
            header,
            text="Elbow Carrying Angle Measurement",
            style="Title.TLabel"
        ).pack(anchor="center")

        ttk.Label(
            header,
            text="Align elbow in front of the camera, capture healthy & injured angles, then download a PDF report.",
            style="Subtitle.TLabel",
            wraplength=1000,
            justify="center"
        ).pack(anchor="center", pady=(2, 10))

        # Video (centered, not stretched)
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(anchor="center", pady=10)

        # Bottom controls bar
        bottom = ttk.Frame(main_frame, style="Bottom.TFrame")
        bottom.pack(fill="x", pady=(10, 10), padx=20)

        # Use a grid with 3 columns to center everything
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=2)
        bottom.columnconfigure(2, weight=1)

        # Center content frame
        center = ttk.Frame(bottom, style="Bottom.TFrame")
        center.grid(row=0, column=1, sticky="n")

        # Patient info row
        patient_frame = ttk.Frame(center, style="Bottom.TFrame")
        patient_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(patient_frame, text="Patient:", style="TLabel").grid(row=0, column=0, padx=3, pady=2, sticky="e")
        self.patient_name_var = tk.StringVar()
        ttk.Entry(patient_frame, textvariable=self.patient_name_var, width=18).grid(row=0, column=1, padx=3, pady=2)

        ttk.Label(patient_frame, text="Age:", style="TLabel").grid(row=0, column=2, padx=3, pady=2, sticky="e")
        self.patient_age_var = tk.StringVar()
        ttk.Entry(patient_frame, textvariable=self.patient_age_var, width=5).grid(row=0, column=3, padx=3, pady=2)

        ttk.Label(patient_frame, text="Sex:", style="TLabel").grid(row=0, column=4, padx=3, pady=2, sticky="e")
        self.patient_sex_var = tk.StringVar()
        sex_combo = ttk.Combobox(
            patient_frame,
            textvariable=self.patient_sex_var,
            values=["", "Male", "Female"],
            width=8,
            state="readonly"
        )
        sex_combo.grid(row=0, column=5, padx=3, pady=2)

        ttk.Label(patient_frame, text="ID:", style="TLabel").grid(row=0, column=6, padx=3, pady=2, sticky="e")
        self.patient_id_var = tk.StringVar()
        ttk.Entry(patient_frame, textvariable=self.patient_id_var, width=12).grid(row=0, column=7, padx=3, pady=2)

        # Settings + capture row
        middle_row = ttk.Frame(center, style="Bottom.TFrame")
        middle_row.pack(fill="x", pady=(5, 5))

        # Injured side selection
        side_frame = ttk.Frame(middle_row, style="Bottom.TFrame")
        side_frame.pack(side="left", padx=(0, 15))

        ttk.Label(side_frame, text="Injured side:", style="TLabel").grid(row=0, column=0, padx=3, pady=2)
        self.injured_side_var = tk.StringVar(value="Right")
        injured_combo = ttk.Combobox(
            side_frame,
            textvariable=self.injured_side_var,
            values=["Right", "Left"],
            state="readonly",
            width=8
        )
        injured_combo.grid(row=0, column=1, padx=3, pady=2)

        # Capture buttons
        btn_frame = ttk.Frame(middle_row, style="Bottom.TFrame")
        btn_frame.pack(side="left")

        ttk.Button(
            btn_frame,
            text="Capture Healthy Side",
            style="Primary.TButton",
            command=self.capture_healthy
        ).grid(row=0, column=0, padx=5, pady=2)

        ttk.Button(
            btn_frame,
            text="Capture Injured Side",
            style="Primary.TButton",
            command=self.capture_injured
        ).grid(row=0, column=1, padx=5, pady=2)

        # PDF button
        pdf_frame = ttk.Frame(middle_row, style="Bottom.TFrame")
        pdf_frame.pack(side="left", padx=(15, 0))

        self.btn_pdf = ttk.Button(
            pdf_frame,
            text="Generate PDF Report",
            style="Primary.TButton",
            command=self.generate_pdf
        )
        self.btn_pdf.grid(row=0, column=0, padx=5, pady=2)
        self.btn_pdf.state(["disabled"])

        # Bottom results / live angles row
        bottom_row = ttk.Frame(center, style="Bottom.TFrame")
        bottom_row.pack(fill="x", pady=(5, 0))

        self.current_angles_label = ttk.Label(
            bottom_row,
            text="Live angles → Right: – degrees   |   Left: – degrees",
            style="TLabel"
        )
        self.current_angles_label.pack()

        self.healthy_label = ttk.Label(
            bottom_row,
            text="Healthy side angle: –",
            style="TLabel"
        )
        self.healthy_label.pack()

        self.injured_label = ttk.Label(
            bottom_row,
            text="Injured side angle: –",
            style="TLabel"
        )
        self.injured_label.pack()

        self.loss_label = ttk.Label(
            bottom_row,
            text="Carrying angle loss: –",
            style="TLabel"
        )
        self.loss_label.pack()

        self.interpret_label = ttk.Label(
            bottom_row,
            text="",
            style="TLabel",
            wraplength=900,
            justify="center"
        )
        self.interpret_label.pack(pady=(2, 0))

    # ---------- CAMERA LOOP ----------

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.after(50, self.update_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(50, self.update_frame)
            return

        self.current_raw_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        right_angle = None
        left_angle = None

        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )

            right_angle = carrying_angle_from_landmarks(
                results.pose_landmarks.landmark, frame.shape, side="right"
            )
            left_angle = carrying_angle_from_landmarks(
                results.pose_landmarks.landmark, frame.shape, side="left"
            )

            # Text on frame with "degrees"
            txt = "Right: "
            txt += f"{right_angle:.1f} degrees" if right_angle is not None else "--"
            txt += "  |  Left: "
            txt += f"{left_angle:.1f} degrees" if left_angle is not None else "--"

            cv2.putText(
                frame,
                txt,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "Elbow not detected. Adjust position.",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        self.current_right_angle = right_angle
        self.current_left_angle = left_angle
        live_txt = "Live angles → Right: "
        live_txt += f"{right_angle:.1f} degrees" if right_angle is not None else "–"
        live_txt += "   |   Left: "
        live_txt += f"{left_angle:.1f} degrees" if left_angle is not None else "–"
        self.current_angles_label.config(text=live_txt)

        self.current_processed_frame = frame.copy()

        # Convert to Tk image - 20% bigger width than before (1000 -> 1200)
        frame_rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb_disp)

        target_width = 1200
        h, w = frame.shape[:2]
        aspect = w / h if h != 0 else 16 / 9
        target_height = int(target_width / aspect)
        img = img.resize((target_width, target_height), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(20, self.update_frame)

    # ---------- CAPTURE BUTTONS ----------

    def capture_healthy(self):
        if self.current_raw_frame is None or self.current_processed_frame is None:
            messagebox.showwarning("No Frame", "No camera frame available yet.")
            return

        injured_side = self.injured_side_var.get()
        healthy_side = "Left" if injured_side == "Right" else "Right"

        if healthy_side == "Right":
            angle = self.current_right_angle
        else:
            angle = self.current_left_angle

        if angle is None:
            messagebox.showwarning(
                "No Angle Detected",
                f"Could not detect a valid carrying angle for the {healthy_side.lower()} elbow.\n\n"
                "Please adjust the position and try again."
            )
            return

        self.healthy_angle = angle
        self.healthy_raw = self.current_raw_frame.copy()
        self.healthy_proc = self.current_processed_frame.copy()

        self.healthy_label.config(
            text=f"Healthy side ({healthy_side}) angle: {self.healthy_angle:.1f} degrees"
        )
        self.update_results_summary()

    def capture_injured(self):
        if self.current_raw_frame is None or self.current_processed_frame is None:
            messagebox.showwarning("No Frame", "No camera frame available yet.")
            return

        injured_side = self.injured_side_var.get()

        if injured_side == "Right":
            angle = self.current_right_angle
        else:
            angle = self.current_left_angle

        if angle is None:
            messagebox.showwarning(
                "No Angle Detected",
                f"Could not detect a valid carrying angle for the {injured_side.lower()} elbow.\n\n"
                "Please adjust the position and try again."
            )
            return

        self.injured_angle = angle
        self.injured_raw = self.current_raw_frame.copy()
        self.injured_proc = self.current_processed_frame.copy()

        self.injured_label.config(
            text=f"Injured side ({injured_side}) angle: {self.injured_angle:.1f} degrees"
        )
        self.update_results_summary()

    def update_results_summary(self):
        if self.healthy_angle is not None and self.injured_angle is not None:
            loss = abs(self.healthy_angle - self.injured_angle)
            self.loss_label.config(text=f"Carrying angle loss: {loss:.1f} degrees")

            # Simple textual interpretation
            if loss <= 5:
                interp = "Side-to-side difference is small and may be within normal variation."
            elif loss <= 10:
                interp = "Moderate difference; correlate with symptoms and clinical examination."
            else:
                interp = "Marked loss of carrying angle; consider detailed clinical and radiographic evaluation."

            self.interpret_label.config(text=interp)
            self.btn_pdf.state(["!disabled"])
        else:
            self.loss_label.config(text="Carrying angle loss: –")
            self.interpret_label.config(text="")
            self.btn_pdf.state(["disabled"])

    # ---------- PDF GENERATION ----------

    def generate_pdf(self):
        if not (
            self.healthy_angle is not None and
            self.injured_angle is not None and
            self.healthy_raw is not None and
            self.healthy_proc is not None and
            self.injured_raw is not None and
            self.injured_proc is not None
        ):
            messagebox.showwarning(
                "Incomplete Data",
                "Please capture both healthy and injured side angles before generating a PDF."
            )
            return

        loss = abs(self.healthy_angle - self.injured_angle)

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="carrying_angle_report.pdf",
            title="Save PDF Report"
        )
        if not save_path:
            return  # user cancelled

        patient_info = {
            "name": self.patient_name_var.get().strip(),
            "age": self.patient_age_var.get().strip(),
            "sex": self.patient_sex_var.get().strip(),
            "id": self.patient_id_var.get().strip(),
        }

        try:
            create_pdf_report(
                self.healthy_angle,
                self.injured_angle,
                loss,
                self.healthy_raw,
                self.healthy_proc,
                self.injured_raw,
                self.injured_proc,
                filename=save_path,
                patient_info=patient_info
            )
            messagebox.showinfo(
                "Report Saved",
                f"PDF report saved to:\n{save_path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to generate PDF report.\n\n{e}"
            )

    # ---------- CLEANUP ----------

    def on_closing(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        try:
            self.pose.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = CarryingAngleApp()
    app.mainloop()