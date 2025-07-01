import cv2
import torch
from ultralytics import YOLO
import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path

# Define distinct colors (BGR) and suggestions for each pathology
COLOR_MAP = {
    "Acne": (0, 0, 255),
    "Blackheads": (0, 255, 0),
    "Dark Circles": (255, 0, 0),
    "Pigmentation": (0, 255, 255),
    "Wrinkles": (255, 0, 255),
    "Pore": (255, 255, 0),
    "Redness": (255, 165, 0),
}

# Confidence thresholds per model
THRESHOLDS = {
    "Acne": 0.6,
    "Blackheads": 0.5,
    "Dark Circles": 0.5,
    "Pigmentation": 0.5,
    "Wrinkles": 0.4,
    "Pore": 0.6,
    "Redness": 0.6,
}

# Suggestions for each pathology
SUGGESTIONS = {
    "Acne": "Consider topical retinoid therapy.",
    "Blackheads": "Recommend salicylic acid exfoliation.",
    "Dark Circles": "Ensure adequate sleep and cold compresses.",
    "Pigmentation": "Use SPF daily and topical lightening agents.",
    "Wrinkles": "Apply daily retinol-based serums.",
    "Pore": "Incorporate clay masks to minimize pore appearance.",
    "Redness": "Use soothing aloe vera or niacinamide products.",
}

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Medical Aesthetic Detector")
        self.geometry("900x700")

        # Locate models directory relative to this script
        repo_root = Path(__file__).resolve().parents[1]
        models_dir = repo_root / "Modeller" / "Medical Aesthetic"

        # Map names to paths
        self.model_paths = {name: str(models_dir / f"{name}.pt") for name in COLOR_MAP}
        self.models = {}
        self.current_model = next(iter(self.model_paths))
        self.load_model(self.current_model)

        # --- Control panel ---
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(side="top", fill="x", pady=5)

        # Model selection buttons
        self.buttons = {}
        for name in self.model_paths:
            btn = ctk.CTkButton(ctrl, text=name, width=100,
                                command=lambda n=name: self.on_model_change(n))
            btn.pack(side="left", padx=2)
            self.buttons[name] = btn
        self.highlight_selected_button()

        # Report button
        self.report_btn = ctk.CTkButton(ctrl, text="Report", command=self.capture_and_analyze)
        self.report_btn.pack(side="right", padx=10)

        # --- Video display ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)

        self.after(10, self.update_frame)

    def load_model(self, name):
        if name not in self.models:
            self.models[name] = YOLO(self.model_paths[name])

    def on_model_change(self, name):
        self.current_model = name
        self.load_model(name)
        self.highlight_selected_button()

    def highlight_selected_button(self):
        for n, btn in self.buttons.items():
            btn.configure(fg_color="#0078D7" if n == self.current_model else "transparent")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            vis = frame.copy()
            # Live inference with model-specific threshold
            thresh = THRESHOLDS[self.current_model]
            results = self.models[self.current_model](frame, conf=thresh)[0]
            color = COLOR_MAP[self.current_model]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, self.current_model, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            vis = cv2.resize(vis, (320, 240))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(vis))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(10, self.update_frame)

    def capture_and_analyze(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        annotated = frame.copy()
        report_lines = []

        # Process all models using their thresholds
        for name, model in self.models.items():
            thresh = THRESHOLDS[name]
            results = model(frame, conf=thresh)[0]
            count = len(results.boxes)
            suggestion = SUGGESTIONS.get(name, "")
            report_lines.append(f"{name}: {count} finding(s). Suggestion: {suggestion}")
            color = COLOR_MAP[name]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Pop-up window with higher-quality image
        popup = ctk.CTkToplevel(self)
        popup.title("Analysis Report")

        # Display annotated image at 640x480
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        lbl = ctk.CTkLabel(popup, image=imgtk, text="")
        lbl.image = imgtk
        lbl.pack(side="left", padx=10, pady=10)

        # Display detailed report
        txt = ctk.CTkTextbox(popup, width=400, height=480)
        txt.insert("1.0", "\n".join(report_lines))
        txt.configure(state="disabled")
        txt.pack(side="right", padx=10, pady=10)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
