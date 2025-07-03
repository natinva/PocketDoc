import sys
import os
import threading
import time
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import math
from tkinter import (
    Tk, Canvas, Button, Label, filedialog, Toplevel,
    CENTER, NW, DISABLED, NORMAL, messagebox
)
from PIL import Image, ImageTk

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
    Raises if required landmarks are missing.
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

    # Overlay angles using "deg" instead of the degree symbol
    cv2.putText(frame,
                f'R: {int(right_angle)} deg',
                tuple(np.multiply(right_elbow, [1, 1]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame,
                f'L: {int(left_angle)} deg',
                tuple(np.multiply(left_elbow, [1, 1]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return frame, left_angle, right_angle


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # — choose proper backend on macOS —
        camera_id = 0
        if sys.platform == 'darwin':
            self.vid = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        else:
            self.vid = cv2.VideoCapture(camera_id)

        # set resolution
        self.width  = 640
        self.height = 480
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # check camera access
        if not self.vid.isOpened():
            messagebox.showerror(
                "Error",
                "❌ Cannot open webcam.\n"
                "Check camera permissions in System Preferences."
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
        self.current_frame       = None
        self.annotated_frame     = None
        self.current_left_angle  = None
        self.current_right_angle = None

        # storage for capture
        self.extension_image      = None
        self.flexion_image        = None
        self.extension_left_angle = 0
        self.extension_right_angle= 0
        self.flexion_left_angle   = 0
        self.flexion_right_angle  = 0

        # — UI setup —
        self.canvas = Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.prompt_label = Label(
            window,
            text="Please fully *extend* your elbow",
            font=("Arial", 12)
        )
        self.prompt_label.pack()

        # initialize with "deg" units
        self.angle_label = Label(
            window,
            text="L: -- deg   R: -- deg",
            font=("Arial", 12)
        )
        self.angle_label.pack()

        # control buttons
        self.btn_stop      = Button(
            window, text="Stop Webcam", width=50,
            command=self.stop_video
        )
        self.btn_image     = Button(
            window, text="Load Image", width=50,
            command=self.load_image
        )
        self.btn_video     = Button(
            window, text="Load Video File", width=50,
            command=self.load_video
        )
        self.btn_extension = Button(
            window, text="Capture Extension", width=50,
            command=self.capture_extension
        )
        self.btn_flexion   = Button(
            window, text="Capture Flexion", width=50,
            command=self.capture_flexion
        )

        for b in (
            self.btn_stop, self.btn_image, self.btn_video,
            self.btn_extension, self.btn_flexion
        ):
            b.pack(anchor=CENTER, expand=True)

        self.btn_flexion.config(state=DISABLED)

        # start periodic UI update & then camera thread
        self.window.after(30, self.periodic_update)
        self.window.after(100, self.start_video)
        self.window.mainloop()

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

            # mirror & resize once
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))

            self.frame_count += 1
            if self.frame_count % 3 == 0:
                processed, lm = detectPose(frame, self.pose)
                if lm:
                    try:
                        processed, l_ang, r_ang = classifyPose(lm, processed)
                        self.current_left_angle  = l_ang
                        self.current_right_angle = r_ang
                    except Exception:
                        self.current_left_angle  = None
                        self.current_right_angle = None
                    self.annotated_frame = processed
                else:
                    self.current_left_angle  = None
                    self.current_right_angle = None
                    self.annotated_frame = processed

            self.current_frame = (
                self.annotated_frame.copy()
                if self.annotated_frame is not None
                else frame
            )
            time.sleep(0.01)

        self.vid.release()
        self.running = False

    def stop_video(self):
        self.running = False

    def periodic_update(self):
        """Refresh canvas and live-angle label every 30ms."""
        if self.current_frame is not None:
            rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.create_image(0, 0, image=img, anchor=NW)
            self.photo = img

            if (self.current_left_angle is not None
                    and self.current_right_angle is not None):
                self.angle_label.config(
                    text=f"L: {self.current_left_angle:.1f} deg   "
                         f"R: {self.current_right_angle:.1f} deg"
                )
            else:
                self.angle_label.config(text="Elbow not detected")

        self.window.after(30, self.periodic_update)

    def load_image(self):
        """Stop video and process a still image."""
        self.stop_video()
        self.vid.release()

        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")]
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        frame = cv2.resize(frame, (self.width, self.height))
        proc, lm = detectPose(frame, self.pose)
        if lm:
            try:
                proc, _, _ = classifyPose(lm, proc)
            except:
                pass

        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, image=img, anchor=NW)
        self.photo = img

    def load_video(self):
        """Stop webcam and process a video file."""
        self.stop_video()
        self.vid.release()

        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files","*.*")]
        )
        if not path:
            return

        self.vid = cv2.VideoCapture(path)
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        self.frame_count = 0
        self.start_video()

    def capture_extension(self):
        """Capture and save the extension frame with timestamp."""
        if self.current_frame is None or self.current_left_angle is None:
            messagebox.showwarning("Warning", "No valid elbow detected.")
            return

        frame = self.current_frame.copy()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"extension_{ts}.png"
        cv2.imwrite(os.path.join(os.getcwd(), fname), frame)

        self.extension_image       = frame
        self.extension_left_angle  = self.current_left_angle
        self.extension_right_angle = self.current_right_angle

        self.prompt_label.config(text="Now fully *flex* your elbow")
        self.btn_extension.config(state=DISABLED)
        self.btn_flexion.config(state=NORMAL)

    def capture_flexion(self):
        """Capture and save the flexion frame, then show comparison."""
        if self.current_frame is None or self.current_left_angle is None:
            messagebox.showwarning("Warning", "No valid elbow detected.")
            return

        frame = self.current_frame.copy()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"flexion_{ts}.png"
        cv2.imwrite(os.path.join(os.getcwd(), fname), frame)

        self.flexion_image        = frame
        self.flexion_left_angle   = self.current_left_angle
        self.flexion_right_angle  = self.current_right_angle

        self.btn_flexion.config(state=DISABLED)
        self.show_side_by_side()

        # reset for next measurement
        self.prompt_label.config(text="Please fully *extend* your elbow")
        self.btn_extension.config(state=NORMAL)

    def show_side_by_side(self):
        """Display a Toplevel window comparing extension vs. flexion."""
        if self.extension_image is None or self.flexion_image is None:
            return

        win = Toplevel(self.window)
        win.title("Extension ⇆ Flexion")

        combined = np.hstack((self.extension_image, self.flexion_image))
        sw = self.window.winfo_screenwidth()
        ratio = sw / combined.shape[1]
        nh = int(combined.shape[0] * ratio)
        resized = cv2.resize(combined, (sw, nh))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        c = Canvas(win, width=sw, height=nh)
        c.pack()
        c.create_image(0, 0, image=img, anchor=NW)
        c.img = img

        ld = abs(self.extension_left_angle - self.flexion_left_angle)
        rd = abs(self.extension_right_angle - self.flexion_right_angle)
        text = (f"Left elbow Δ = {ld:.1f} deg"
                if ld > rd else
                f"Right elbow Δ = {rd:.1f} deg")

        Label(win, text=text, font=("Arial", 14)).pack()
        win.mainloop()


if __name__ == "__main__":
    root = Tk()
    App(root, "Elbow ROM Measurement")
