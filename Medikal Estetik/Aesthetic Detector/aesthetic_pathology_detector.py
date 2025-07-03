import cv2
import torch
from ultralytics import YOLO
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
import tempfile
import os

# ------------ Configuration ------------
BTN_BG = "#003366"
BTN_SELECTED = "#002244"
BTN_HOVER = "#004080"
BTN_TEXT = "#e6f2ff"

COLOR_MAP = {
    "Acne": (0, 0, 255),
    "Blackheads": (0, 255, 0),
    "Dark Circles": (255, 0, 0),
    "Pigmentation": (0, 255, 255),
    "Wrinkles": (255, 0, 255),
    "Pore": (255, 255, 0),
    "Redness": (255, 165, 0),
}
THRESHOLDS = {"Acne":0.6, "Blackheads":0.5, "Dark Circles":0.5,
              "Pigmentation":0.5, "Wrinkles":0.4, "Pore":0.6, "Redness":0.6}
DROP = {"Acne":9, "Blackheads":8, "Dark Circles":14,
        "Pigmentation":23, "Wrinkles":18, "Pore":17, "Redness":27}
EXPLANATIONS = {
    "Acne":"Acne is caused by clogged pores, bacteria, and excess oil production.",
    "Blackheads":"Blackheads form when pores are partially clogged with oil and dead skin cells.",
    "Dark Circles":"Dark circles can be due to genetics, thin under-eye skin, or hyperpigmentation.",
    "Pigmentation":"Pigmentation irregularities arise from melanin overproduction, triggered by sun exposure.",
    "Wrinkles":"Wrinkles develop from loss of collagen and elastin over time.",
    "Pore":"Enlarged pores result from genetics, oiliness, and loss of skin elasticity.",
    "Redness":"Redness can be caused by irritation, inflammation, or vascular issues.",
}
SUGGESTIONS = {
    "Acne":"Use a gentle salicylic acid wash and consult a dermatologist.",
    "Blackheads":"Try chemical exfoliation with BHA (salicylic acid).",
    "Dark Circles":"Ensure sufficient sleep and use cold compresses.",
    "Pigmentation":"Apply daily SPF 30+ and use topical lightening agents.",
    "Wrinkles":"Incorporate retinoids at night and protect skin from UV.",
    "Pore":"Use clay masks weekly and oil-control primers.",
    "Redness":"Choose fragrance-free, soothing products.",
}

# Determine quality label
def quality_label(pct):
    if pct > 80:
        return "Good"
    elif pct >= 40:
        return "Neutral"
    else:
        return "Poor"
# ---------------------------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Medical Aesthetic Detector")
        self.geometry("900x700")

        # Register custom font for PDF
        repo = Path(__file__).resolve().parents[2]
        font_path = repo / "Fonts/League_Spartan/static/LeagueSpartan-SemiBold.ttf"
        pdfmetrics.registerFont(TTFont("LeagueSpartan", str(font_path)))

        # Load models
        models_dir = repo / "Modeller" / "Medical Aesthetic"
        file_names = {
            "Acne":"acne.pt","Blackheads":"Blackheads.pt","Dark Circles":"Dark Circles.pt",
            "Pigmentation":"Pigmentation.pt","Wrinkles":"Wrinkles.pt","Pore":"pore.pt","Redness":"redness.pt"
        }
        self.models = {name: YOLO(str(models_dir / fname)) for name, fname in file_names.items()}
        self.current_model = next(iter(self.models))

        # Control panel
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(side="top", fill="x", pady=5)
        self.buttons = {}
        for name in self.models:
            btn = ctk.CTkButton(
                ctrl, text=name, width=120,
                fg_color=BTN_BG, hover_color=BTN_HOVER, text_color=BTN_TEXT,
                command=lambda n=name: self.on_model_change(n)
            )
            btn.pack(side="left", padx=4)
            self.buttons[name] = btn
        self.highlight_selected_button()

        # Live video display
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.video_label = tk.Label(self)
        self.video_label.pack(pady=10)
        self.after(10, self.update_frame)

        # Report button
        self.report_btn = ctk.CTkButton(
            self, text="Report", width=200,
            fg_color=BTN_BG, hover_color=BTN_HOVER, text_color=BTN_TEXT,
            command=self.capture_and_analyze
        )
        self.report_btn.pack(pady=10)

    def on_model_change(self, name):
        self.current_model = name
        self.highlight_selected_button()

    def highlight_selected_button(self):
        for name, btn in self.buttons.items():
            btn.configure(fg_color=BTN_SELECTED if name == self.current_model else BTN_BG)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            vis = frame.copy()
            results = self.models[self.current_model](frame, conf=THRESHOLDS[self.current_model])[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_MAP[self.current_model], 2)
                cv2.putText(vis, self.current_model, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP[self.current_model], 1)
            img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(10, self.update_frame)

    def capture_and_analyze(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        annotated = frame.copy()
        report_data = []
        # Process every model
        for name, model in self.models.items():
            results = model(frame, conf=THRESHOLDS[name])[0]
            count = len(results.boxes)
            pct = max(0, 100 - DROP[name] * count)
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_MAP[name], 2)
            quality = quality_label(pct)
            report_data.append((name, quality, EXPLANATIONS[name], SUGGESTIONS[name]))

        # Pop-up report
        popup = ctk.CTkToplevel(self)
        popup.title("Analysis Report")
        popup.geometry("800x900")

        # Display annotated image
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img).resize((640, 480))
        imgtk = ImageTk.PhotoImage(pil_img)
        tk.Label(popup, image=imgtk).pack(pady=10)
        popup.img = imgtk  # prevent GC

        # Scrollable frame
        container = tk.Frame(popup)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Populate report sections
        for name, quality, expl, sugg in report_data:
            tk.Label(frame, text=f"{name} - {quality}", font=(None, 14, 'bold')).pack(anchor='w', padx=20)
            tk.Label(frame, height=1).pack()
            bar = ctk.CTkProgressBar(frame, width=600, fg_color="#cccccc", progress_color=BTN_BG)
            # Set bar fraction: Good=1, Neutral=.6, Poor=.2
            frac = {'Good':1.0, 'Neutral':0.6, 'Poor':0.2}[quality]
            bar.set(frac)
            bar.pack(pady=5, padx=20)
            tk.Label(frame, height=1).pack()
            tk.Label(frame, text=f"Explanation: {expl}", wraplength=700, justify='left').pack(anchor='w', padx=20)
            tk.Label(frame, text=f"Suggestion: {sugg}", wraplength=700, justify='left').pack(anchor='w', padx=20)
            for _ in range(2): tk.Label(frame, height=1).pack()

        # Download PDF button
        pdf_btn = ctk.CTkButton(
            popup, text="Download PDF", width=200,
            fg_color=BTN_BG, hover_color=BTN_HOVER, text_color=BTN_TEXT,
            command=lambda: self.generate_pdf(annotated, report_data)
        )
        pdf_btn.pack(pady=10)

    def generate_pdf(self, image, data):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = Path.cwd() / f"Report_{ts}.pdf"
        # Save annotated image
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(tmp.name, image)
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        w, h = A4
        img_w, img_h = 400, 300
        x, y = (w - img_w) / 2, h - img_h - 50
        # Draw image once at top of first page
        c.drawImage(ImageReader(tmp.name), x, y, img_w, img_h)
        # Draw each model sequentially below
        text_y = y - 30
        for name, quality, expl, sugg in data:
            # Draw header
            c.setFont("LeagueSpartan", 12)
            c.drawString(50, text_y, f"{name} - {quality}")
            text_y -= 20
            # Draw bar
            frac = {'Good':1.0, 'Neutral':0.6, 'Poor':0.2}[quality]
            bar_len = frac * (w - 100)
            c.setFillColor(HexColor(BTN_BG))
            c.rect(50, text_y - 10, bar_len, 15, fill=1, stroke=0)
            text_y -= 30
            # Draw explanation and suggestion
            c.setFillColor(HexColor("#000000"))
            c.setFont("LeagueSpartan", 10)
            c.drawString(50, text_y, f"Explanation: {expl}")
            text_y -= 18
            c.drawString(50, text_y, f"Suggestion: {sugg}")
            text_y -= 30
            # Check if we need a new page
            if text_y < 80:
                c.showPage()
                c.drawImage(ImageReader(tmp.name), x, y, img_w, img_h)
                text_y = y - 30
        c.save()
        tmp.close()
        os.unlink(tmp.name)
        messagebox.showinfo("Saved", f"PDF saved as {pdf_path}")

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
