import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import cv2
import mediapipe as mp
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Determine script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Register League Spartan font for PDF
FONT_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'Fonts', 'League_Spartan', 'static', 'LeagueSpartan-SemiBold.ttf')
pdfmetrics.registerFont(TTFont('LeagueSpartan', FONT_PATH))

# Normative means and SDs for angle comparisons
NORMS = {
    'Nasofrontal Angle': (138.5, 6.5),
    'Nasolabial Angle': (104.0, 10.0),
    'Nasomental Angle': (125.0, 8.0),
    'Nasofacial Angle': (10.0, 5.0),
}

# Interpretations for each measurement
INTERPRETATIONS = {
    'Nasofrontal Angle': 'Slope of forehead to nasal root: values above normal indicate dorsal hump prominence.',
    'Nasolabial Angle': 'Tip rotation: above normal suggests over-rotated tip; below normal suggests ptotic tip.',
    'Nasomental Angle': 'Chin projection relative to nasal tip: higher values indicate retruded chin.',
    'Nasofacial Angle': 'Nasal projection relative to facial plane: values above normal indicate convex profile.',
}

mp_face_mesh = mp.solutions.face_mesh

# Helper to calculate angle between three points
def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ab = a - b
    cb = c - b
    dot = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return None
    return np.degrees(np.arccos(dot / (norm_ab * norm_cb)))

# Core processing with selective angles and optional progress callback
def analyze_lateral_angles(image_path, selected_angles, progress_callback=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image.")
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if progress_callback:
        progress_callback('start')

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            if progress_callback:
                progress_callback('stop')
            raise ValueError("No face landmarks detected.")

        landmarks = [(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark]
        pts = lambda idx: tuple(map(int, landmarks[idx]))
        pts_map = {
            'forehead': pts(10),
            'nose_bridge': pts(168),
            'nose_tip': pts(1),
            'nasal_root': pts(376),
            'chin_tip': pts(152)
        }

        raw = {
            'Nasofrontal Angle': calculate_angle(pts_map['forehead'], pts_map['nose_bridge'], pts_map['nose_tip']),
            'Nasolabial Angle': calculate_angle(pts_map['nose_tip'], pts_map['nasal_root'], pts_map['chin_tip']),
            'Nasomental Angle': calculate_angle(pts_map['nose_bridge'], pts_map['nose_tip'], pts_map['chin_tip']),
            'Nasofacial Angle': None
        }
        if raw['Nasofrontal Angle'] is not None:
            raw['Nasofacial Angle'] = 180 - raw['Nasofrontal Angle']

        # Draw selected angles on image
        for name, val in raw.items():
            if name not in selected_angles or val is None:
                continue
            p = pts_map
            if name == 'Nasofrontal Angle':
                cv2.line(image, p['forehead'], p['nose_bridge'], (0,255,0), 2)
                cv2.line(image, p['nose_bridge'], p['nose_tip'], (0,255,0), 2)
            elif name == 'Nasolabial Angle':
                cv2.line(image, p['nose_tip'], p['nasal_root'], (0,255,0), 2)
                cv2.line(image, p['nasal_root'], p['chin_tip'], (0,255,0), 2)
            elif name == 'Nasomental Angle':
                cv2.line(image, p['nose_tip'], p['chin_tip'], (0,255,0), 2)
            elif name == 'Nasofacial Angle':
                cv2.line(image, p['forehead'], p['chin_tip'], (0,255,0), 2)

        # Overlay text with 'degrees' to avoid unsupported characters
        y_text = 30
        for name, val in raw.items():
            if name not in selected_angles:
                continue
            cv2.putText(image,
                        f"{name}: {val:.2f} degrees",
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        2)
            y_text += 30

        # Save processed image
        base, ext = os.path.splitext(image_path)
        out = f"{base}_processed{ext}"
        cv2.imwrite(out, image)

    if progress_callback:
        progress_callback('stop')

    return {n: raw[n] for n in selected_angles}, out

class FaceAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Profiloplasty Angle Analysis")
        self.root.geometry("900x700")

        tk.Label(root, text="Profiloplasty Face Analysis", font=("Arial", 18, 'bold')).pack(pady=10)
        ctrl = tk.Frame(root)
        ctrl.pack(pady=5)

        self.upload_btn = tk.Button(ctrl, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=5)

        self.angle_vars = {}
        for i, name in enumerate(NORMS.keys(), start=1):
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(ctrl, text=name, var=var).grid(row=0, column=i, padx=5)
            self.angle_vars[name] = var

        self.progress = ttk.Progressbar(root, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=10)

        self.frame = tk.Frame(root)
        self.frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.img_label = tk.Label(self.frame)
        self.img_label.pack(side='left', padx=10)

        self.txt = tk.Text(self.frame, width=40, state='disabled', font=("Arial", 12))
        self.txt.pack(side='right', padx=10)

        self.pdf_btn = tk.Button(root,
                                  text="Generate PDF Report",
                                  command=self.generate_pdf,
                                  state='disabled')
        self.pdf_btn.pack(pady=5)

        self.current_angles = None
        self.processed_image_path = None

    def progress_callback(self, status):
        if status == 'start':
            self.progress.start()
        else:
            self.progress.stop()

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path:
            return
        selected = [n for n, v in self.angle_vars.items() if v.get()]
        try:
            angles, out = analyze_lateral_angles(path, selected, self.progress_callback)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.processed_image_path = out
        self.current_angles = angles
        self.display_results()
        self.pdf_btn.config(state='normal')

    def display_results(self):
        img = Image.open(self.processed_image_path)
        img = img.resize((400, 400))
        tkimg = ImageTk.PhotoImage(img)
        self.img_label.config(image=tkimg)
        self.img_label.image = tkimg

        self.txt.config(state='normal')
        self.txt.delete('1.0', tk.END)
        for name, val in self.current_angles.items():
            self.txt.insert(tk.END, f"{name}: {val:.2f}°\n")
        self.txt.config(state='disabled')

    def generate_pdf(self):
        if not self.current_angles:
            return
        report_name = os.path.splitext(os.path.basename(self.processed_image_path))[0] + "_report.pdf"
        save_path = os.path.join(SCRIPT_DIR, report_name)

        c = canvas.Canvas(save_path)
        width, height = c._pagesize
        margin = 50

        # Title
        c.setFont('LeagueSpartan', 16)
        c.drawCentredString(width/2, height - margin, "Profiloplasty Angle Analysis Report")

        # Image
        img_w, img_h = 300, 300
        x_img = (width - img_w) / 2
        y_img = height - margin - img_h - 20
        c.drawImage(self.processed_image_path,
                    x_img, y_img,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True)

        # Measurements with normals and interpretations
        c.setFont('LeagueSpartan', 12)
        y_text = y_img - 30
        for name, val in self.current_angles.items():
            mean, sd = NORMS[name]
            normal_range = f"{mean:.1f}° ± {sd:.1f}°"
            interp = INTERPRETATIONS[name]
            c.drawString(margin, y_text, f"{name}: {val:.2f}°")
            y_text -= 18
            c.drawString(margin, y_text, f"Normal: {normal_range}")
            y_text -= 18
            c.drawString(margin, y_text, interp)
            y_text -= 30

        c.save()
        messagebox.showinfo("Saved", f"PDF report automatically saved to {save_path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceAnalysisApp(root)
    root.mainloop()
