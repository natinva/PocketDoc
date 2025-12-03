import sys
import os
import threading
import time
from datetime import datetime
import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from tkinter import (
    Tk, Canvas, Button, Label, filedialog, Toplevel,
    CENTER, NW, DISABLED, NORMAL, messagebox,
    Frame, Entry
)
from PIL import Image, ImageTk

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

# Pose setup using Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b formed by points a–b–c.
    a, b, c: (x, y) pairs.
    Returns angle in degrees [0, 180].
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) \
            - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def detectPose(frame, pose):
    """
    Run Mediapipe pose detection.
    Returns annotated frame and list of landmarks (or None).
    """
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)
        return frame, results.pose_landmarks.landmark
    return frame, None


def classifyPose(landmarks, frame):
    """
    Compute left/right elbow angles (with mirror fix) and overlay them.
    Returns (annotated_frame, left_angle, right_angle).
    """
    # Mirror-corrected: LEFT_* are actually right side in selfie view
    rs = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    re = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    rw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    ls = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    le = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    lw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    h, w, _ = frame.shape
    right_shoulder = [rs.x * w, rs.y * h]
    right_elbow    = [re.x * w, re.y * h]
    right_wrist    = [rw.x * w, rw.y * h]
    left_shoulder  = [ls.x * w, ls.y * h]
    left_elbow     = [le.x * w, le.y * h]
    left_wrist     = [lw.x * w, lw.y * h]

    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_angle  = calculate_angle(left_shoulder,  left_elbow,  left_wrist)

    # Overlay angles using "deg"
    cv2.putText(frame,
                f'R: {int(right_angle)} deg',
                tuple(np.multiply(right_elbow, [1, 1]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame,
                f'L: {int(left_angle)} deg',
                tuple(np.multiply(left_elbow, [1, 1]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, left_angle, right_angle


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#e6f2ff")
        self.window.geometry("1100x800")

        # — choose proper backend on macOS —
        camera_id = 0
        if sys.platform == 'darwin':
            self.vid = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        else:
            self.vid = cv2.VideoCapture(camera_id)

        # base capture resolution (will be scaled to window)
        self.width  = 640
        self.height = 480
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # check camera access
        if not self.vid.isOpened():
            messagebox.showerror(
                "Error",
                "❌ Cannot open webcam.\n"
                "Check camera permissions."
            )
            window.destroy()
            return

        # Mediapipe pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            model_complexity=1
        )

        # shared state
        self.running             = False
        self.frame_count         = 0
        self.current_frame       = None       # what we show on canvas
        self.annotated_frame     = None       # processed frame with overlays
        self.raw_frame           = None       # raw frame (no overlays)
        self.current_left_angle  = None
        self.current_right_angle = None

        # smoothing buffers (last 5 frames)
        self.left_history  = deque(maxlen=5)
        self.right_history = deque(maxlen=5)

        # storage for capture (raw and processed)
        self.extension_raw_image   = None
        self.extension_proc_image  = None
        self.flexion_raw_image     = None
        self.flexion_proc_image    = None
        self.extension_left_angle  = 0.0
        self.extension_right_angle = 0.0
        self.flexion_left_angle    = 0.0
        self.flexion_right_angle   = 0.0

        # === UI ===

        # Big video canvas (slightly lighter blue)
        self.canvas = Canvas(
            window,
            bg="#004b80",
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Metrics bar directly under video
        metrics_frame = Frame(window, bg="#e6f2ff")
        metrics_frame.pack(fill="x", pady=(0, 5))

        self.angle_label = Label(
            metrics_frame,
            text="L: -- deg   R: -- deg",
            bg="#e6f2ff",
            fg="#003366",
            font=("Arial", 12)
        )
        self.angle_label.grid(row=0, column=1, padx=10, pady=3, sticky="e")

        self.rom_label = Label(
            metrics_frame,
            text="ROM: -- deg",
            bg="#e6f2ff",
            fg="#003366",
            font=("Arial", 12, "bold")
        )
        self.rom_label.grid(row=0, column=2, padx=10, pady=3, sticky="w")

        # Center metrics by giving side columns weight
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(3, weight=1)

        # Bottom control bar
        control_frame = Frame(window, bg="#004b80")
        control_frame.pack(fill="x", pady=(0, 0))

        # Patient info – row 0
        lbl_style = {"bg": "#004b80", "fg": "white", "font": ("Arial", 10)}

        Label(control_frame, text="Name:", **lbl_style).grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.entry_name = Entry(control_frame, width=18)
        self.entry_name.grid(row=0, column=1, padx=4, pady=4)

        Label(control_frame, text="Age:", **lbl_style).grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.entry_age = Entry(control_frame, width=5)
        self.entry_age.grid(row=0, column=3, padx=4, pady=4)

        Label(control_frame, text="Sex:", **lbl_style).grid(row=0, column=4, sticky="e", padx=4, pady=4)
        self.entry_sex = Entry(control_frame, width=8)
        self.entry_sex.grid(row=0, column=5, padx=4, pady=4)

        Label(control_frame, text="Patient ID:", **lbl_style).grid(row=0, column=6, sticky="e", padx=4, pady=4)
        self.entry_id = Entry(control_frame, width=12)
        self.entry_id.grid(row=0, column=7, padx=4, pady=4)

        # Prompt – row 1, centred across all columns
        self.prompt_label = Label(
            control_frame,
            text="Please fully *extend* your elbow",
            bg="#004b80",
            fg="#e6f2ff",
            font=("Arial", 11)
        )
        self.prompt_label.grid(row=1, column=0, columnspan=8, pady=(2, 4))

        # Buttons – row 2, centred
        btn_style = {
            "bg": "#e6f2ff",
            "fg": "#004b80",
            "activebackground": "#ffffff",
            "activeforeground": "#004b80",
            "font": ("Arial", 11, "bold"),
            "bd": 0,
            "highlightthickness": 0,
            "padx": 10,
            "pady": 4
        }

        self.btn_extension = Button(
            control_frame,
            text="Capture Extension",
            command=self.capture_extension,
            **btn_style
        )
        self.btn_extension.grid(row=2, column=2, columnspan=2, padx=5, pady=6, sticky="e")

        self.btn_flexion = Button(
            control_frame,
            text="Capture Flexion",
            command=self.capture_flexion,
            state=DISABLED,
            **btn_style
        )
        self.btn_flexion.grid(row=2, column=4, columnspan=2, padx=5, pady=6, sticky="w")

        # Make columns flexible so layout stays centred
        for col in range(8):
            control_frame.grid_columnconfigure(col, weight=1)

        # start periodic UI update & camera thread
        self.window.after(30, self.periodic_update)
        self.window.after(100, self.start_video)
        self.window.mainloop()

    # ---------- video & scaling ----------

    def start_video(self):
        """Spawn worker thread to grab/process frames."""
        if getattr(self, "video_thread", None) and self.video_thread.is_alive():
            return
        self.running = True
        self.video_thread = threading.Thread(
            target=self.video_loop, daemon=True
        )
        self.video_thread.start()

    def video_loop(self):
        """Continuously capture frames, process every 3rd, and store results."""
        while self.running:
            ret, frame = self.vid.read()
            if not ret:
                break

            # mirror & resize once (base resolution)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))

            # keep a raw copy (no overlays)
            self.raw_frame = frame.copy()

            self.frame_count += 1
            if self.frame_count % 3 == 0:
                processed, lm = detectPose(frame, self.pose)
                if lm:
                    try:
                        processed, l_ang, r_ang = classifyPose(lm, processed)

                        # smoothing
                        self.left_history.append(l_ang)
                        self.right_history.append(r_ang)

                        if self.left_history and self.right_history:
                            self.current_left_angle  = float(np.median(self.left_history))
                            self.current_right_angle = float(np.median(self.right_history))
                        else:
                            self.current_left_angle  = None
                            self.current_right_angle = None
                    except Exception:
                        self.current_left_angle  = None
                        self.current_right_angle = None
                        self.left_history.clear()
                        self.right_history.clear()

                    self.annotated_frame = processed
                else:
                    self.current_left_angle  = None
                    self.current_right_angle = None
                    self.left_history.clear()
                    self.right_history.clear()
                    self.annotated_frame = processed

            self.current_frame = (
                self.annotated_frame.copy()
                if self.annotated_frame is not None
                else self.raw_frame
            )
            time.sleep(0.01)

        self.vid.release()
        self.running = False

    def get_scaled_frame(self, frame):
        """
        Scale frame to fit the canvas while maintaining aspect ratio.
        """
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return frame  # window not ready yet

        h, w, _ = frame.shape
        scale = min(canvas_w / w, canvas_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        return resized

    def periodic_update(self):
        """Refresh canvas and live-angle label every 30ms."""
        if self.current_frame is not None:
            scaled = self.get_scaled_frame(self.current_frame)
            rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))

            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=img,
                anchor=CENTER
            )
            self.photo = img

            if (self.current_left_angle is not None
                    and self.current_right_angle is not None):
                self.angle_label.config(
                    text=f"L: {self.current_left_angle:.1f} deg   "
                         f"R: {self.current_right_angle:.1f} deg"
                )
            else:
                self.angle_label.config(text="L: -- deg   R: -- deg")

        self.window.after(30, self.periodic_update)

    # ---------- capture logic (extension / flexion) ----------

    def capture_extension(self):
        """Capture extension (raw + processed) and save angles."""
        if (self.raw_frame is None or
                self.current_left_angle is None or
                self.current_right_angle is None):
            messagebox.showwarning("Warning", "No valid elbow detected.")
            return

        raw = self.raw_frame.copy()
        proc = (self.annotated_frame.copy()
                if self.annotated_frame is not None
                else raw.copy())

        self.extension_raw_image   = raw
        self.extension_proc_image  = proc
        self.extension_left_angle  = self.current_left_angle
        self.extension_right_angle = self.current_right_angle

        self.prompt_label.config(text="Now fully *flex* your elbow")
        self.btn_extension.config(state=DISABLED)
        self.btn_flexion.config(state=NORMAL)

    def capture_flexion(self):
        """
        Capture flexion (raw + processed), compute ROM from the side
        with the larger change, show comparison, and generate PDF.
        """
        if (self.raw_frame is None or
                self.current_left_angle is None
                or self.current_right_angle is None):
            messagebox.showwarning("Warning", "No valid elbow detected.")
            return

        if self.extension_raw_image is None or self.extension_proc_image is None:
            messagebox.showwarning("Warning", "Please capture extension first.")
            return

        raw = self.raw_frame.copy()
        proc = (self.annotated_frame.copy()
                if self.annotated_frame is not None
                else raw.copy())

        self.flexion_raw_image     = raw
        self.flexion_proc_image    = proc
        self.flexion_left_angle    = self.current_left_angle
        self.flexion_right_angle   = self.current_right_angle

        # compute ROM based on side with larger delta
        ld = abs(self.extension_left_angle  - self.flexion_left_angle)
        rd = abs(self.extension_right_angle - self.flexion_right_angle)

        if ld >= rd:
            side_label = "Left"
            ext_angle = self.extension_left_angle
            flex_angle = self.flexion_left_angle
            rom = ld
        else:
            side_label = "Right"
            ext_angle = self.extension_right_angle
            flex_angle = self.flexion_right_angle
            rom = rd

        # update ROM label in metrics bar
        self.rom_label.config(text=f"ROM ({side_label}): {rom:.1f} deg")

        # comparison window and PDF
        self.show_side_by_side(side_label, rom)
        self.create_pdf_report(side_label, ext_angle, flex_angle, rom)

        # reset for next measurement
        self.prompt_label.config(text="Please fully *extend* your elbow")
        self.btn_flexion.config(state=DISABLED)
        self.btn_extension.config(state=NORMAL)

    # ---------- outputs: side-by-side window & PDF ----------

    def show_side_by_side(self, side_label, rom):
        """Display a Toplevel window comparing extension vs. flexion (processed images)."""
        if (self.extension_proc_image is None or
                self.flexion_proc_image is None):
            return

        win = Toplevel(self.window)
        win.title("Extension ⇆ Flexion")

        combined = np.hstack((self.extension_proc_image,
                              self.flexion_proc_image))

        sw = self.window.winfo_screenwidth()
        ratio = sw / combined.shape[1]
        nh = int(combined.shape[0] * ratio)
        resized = cv2.resize(combined, (sw, nh))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        c = Canvas(win, width=sw, height=nh,
                   bg="#004b80", highlightthickness=0, bd=0)
        c.pack()
        c.create_image(0, 0, image=img, anchor=NW)
        c.img = img

        Label(win,
              text=f"{side_label} elbow ROM = {rom:.1f} deg",
              font=("Arial", 14),
              bg="#004b80",
              fg="#e6f2ff").pack(fill="x")

    def create_pdf_report(self, side_label, ext_angle, flex_angle, rom):
        """
        Generate a single-page PDF report:
        - Top: extension & flexion images (raw + processed)
        - Bottom: explanation text (angles, ROM, what ROM means, typical values)
        """
        if (self.extension_raw_image is None or
                self.extension_proc_image is None or
                self.flexion_raw_image is None or
                self.flexion_proc_image is None):
            messagebox.showwarning("Warning", "Missing images for PDF report.")
            return

        # ask user where to save
        default_name = datetime.now().strftime("Elbow_ROM_Report_%Y%m%d_%H%M%S.pdf")
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not path:
            return  # user cancelled

        patient_name = self.entry_name.get().strip()
        patient_age  = self.entry_age.get().strip()
        patient_sex  = self.entry_sex.get().strip()
        patient_id   = self.entry_id.get().strip()

        c = pdf_canvas.Canvas(path, pagesize=A4)
        pw, ph = A4
        margin = 40

        # Title & patient info
        y = ph - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, "Elbow Range of Motion (ROM) Report")
        y -= 20

        c.setFont("Helvetica", 9)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawString(margin, y, f"Date / Time: {ts}")
        y -= 12

        if patient_name:
            c.drawString(margin, y, f"Name: {patient_name}")
            y -= 12
        if patient_age:
            c.drawString(margin, y, f"Age: {patient_age}")
            y -= 12
        if patient_sex:
            c.drawString(margin, y, f"Sex: {patient_sex}")
            y -= 12
        if patient_id:
            c.drawString(margin, y, f"Patient ID: {patient_id}")
            y -= 12

        c.drawString(margin, y, f"Measured side (largest change): {side_label}")
        y -= 12
        c.drawString(margin, y, f"Extension angle: {ext_angle:.1f} deg")
        y -= 12
        c.drawString(margin, y, f"Flexion angle:   {flex_angle:.1f} deg")
        y -= 12
        c.drawString(margin, y, f"ROM (flexion - extension): {rom:.1f} deg")
        y -= 20

        # Image block area at the top half of the page
        top_block_height = ph * 0.5
        img_top_y = y

        # 2x2 grid: [Ext raw] [Ext processed] / [Flex raw] [Flex processed]
        max_w = (pw - 2 * margin) / 2.0 - 10
        max_h = (top_block_height - 30) / 2.0

        def pil_from_cv(bgr_img):
            return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

        def scaled_size(pil_img):
            w, h = pil_img.size
            scale = min(max_w / w, max_h / h)
            return int(w * scale), int(h * scale)

        ext_raw_pil  = pil_from_cv(self.extension_raw_image)
        ext_proc_pil = pil_from_cv(self.extension_proc_image)
        flex_raw_pil = pil_from_cv(self.flexion_raw_image)
        flex_proc_pil= pil_from_cv(self.flexion_proc_image)

        er_w, er_h = scaled_size(ext_raw_pil)
        ep_w, ep_h = scaled_size(ext_proc_pil)
        fr_w, fr_h = scaled_size(flex_raw_pil)
        fp_w, fp_h = scaled_size(flex_proc_pil)

        col1_x = margin
        col2_x = margin + max_w + 20

        row1_y = img_top_y - er_h
        row2_y = row1_y - fr_h - 10

        # Draw images
        c.drawImage(ImageReader(ext_raw_pil),
                    col1_x, row1_y,
                    width=er_w, height=er_h)
        c.drawImage(ImageReader(ext_proc_pil),
                    col2_x, row1_y,
                    width=ep_w, height=ep_h)
        c.drawImage(ImageReader(flex_raw_pil),
                    col1_x, row2_y,
                    width=fr_w, height=fr_h)
        c.drawImage(ImageReader(flex_proc_pil),
                    col2_x, row2_y,
                    width=fp_w, height=fp_h)

        # labels for images
        c.setFont("Helvetica", 8)
        c.drawString(col1_x, row1_y + er_h + 4, "Extension - Unprocessed")
        c.drawString(col2_x, row1_y + ep_h + 4, "Extension - Processed")
        c.drawString(col1_x, row2_y + fr_h + 4, "Flexion - Unprocessed")
        c.drawString(col2_x, row2_y + fp_h + 4, "Flexion - Processed")

        # Explanation text below image block
        text_start_y = row2_y - 40
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, text_start_y, "Explanation:")
        text_start_y -= 14

        c.setFont("Helvetica", 9)
        text_obj = c.beginText()
        text_obj.setTextOrigin(margin, text_start_y)
        text_obj.textLines([
            "Elbow ROM (range of motion) is the difference between maximal flexion and maximal extension of the elbow joint.",
            "In healthy adults, a typical arc is approximately 0° (full extension) to 140–150° of flexion.",
            "For most daily activities, a functional arc of about 30°–130° is usually sufficient.",
            "",
            f"In this measurement, the selected side ({side_label}) showed a range of motion of {rom:.1f} degrees.",
            "This automated measurement is intended for documentation and clinical decision support.",
            "Final interpretation and treatment decisions must be made by a qualified healthcare professional."
        ])
        c.drawText(text_obj)

        c.save()
        messagebox.showinfo("Report saved", f"PDF report saved to:\n{path}")


if __name__ == "__main__":
    root = Tk()
    App(root, "Elbow ROM Measurement")
