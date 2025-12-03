import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ExifTags
import os
import cv2
import mediapipe as mp
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

# ---------------- PATHS & FONT REGISTRATION ----------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_PATH = os.path.join(
    SCRIPT_DIR,
    '..', '..',
    'Fonts',
    'League_Spartan',
    'static',
    'LeagueSpartan-SemiBold.ttf'
)

try:
    pdfmetrics.registerFont(TTFont('LeagueSpartan', FONT_PATH))
    PDF_FONT = 'LeagueSpartan'
except Exception:
    PDF_FONT = 'Helvetica'

# ---------------- NORMATIVE DATA & INTERPRETATIONS ----------------

NORMS = {
    'Nasofrontal Angle': (138.5, 6.5),
    'Nasolabial Angle': (104.0, 10.0),
    'Nasomental Angle': (125.0, 8.0),
    'Nasofacial Angle': (10.0, 5.0),
}

INTERPRETATIONS = {
    'Nasofrontal Angle': 'Slope of forehead to nasal root: values above normal may indicate dorsal hump.',
    'Nasolabial Angle': 'Tip rotation: above normal suggests over-rotated tip; below normal suggests ptotic tip.',
    'Nasomental Angle': 'Chin projection relative to nasal tip: higher values may indicate retruded chin.',
    'Nasofacial Angle': 'Nasal projection relative to facial plane.',
}

mp_face_mesh = mp.solutions.face_mesh

# ---------------- GEOMETRY HELPERS ----------------

def calculate_angle(point1, point2, point3):
    a = np.array(point1, dtype=float)
    b = np.array(point2, dtype=float)
    c = np.array(point3, dtype=float)
    ab = a - b
    cb = c - b
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return None
    cos_val = np.dot(ab, cb) / (norm_ab * norm_cb)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def angle_between_segments(p1, p2, q1, q2):
    v1 = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    v2 = np.array(q2, dtype=float) - np.array(q1, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_val = np.dot(v1, v2) / (n1 * n2)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def classify_deviation(value, mean, sd):
    if sd <= 0:
        return "No normative data"
    z = (value - mean) / sd
    az = abs(z)
    if az < 1:
        return "Within normal range"
    elif az < 2:
        return "Mild deviation"
    else:
        return "Marked deviation"


def auto_rotate_pil_image(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if not exif:
            return img
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == 'Orientation'),
            None
        )
        if not orientation_key or orientation_key not in exif:
            return img
        orientation = exif[orientation_key]
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        return img
    return img


def limit_image_size_cv2(image, max_dim=1600, min_dim=600):
    """
    Downscale very large images, upscale very small ones, for more stable FaceMesh.
    """
    h, w = image.shape[:2]
    max_current = max(h, w)
    min_current = min(h, w)

    scale = 1.0
    if max_current > max_dim:
        scale = max_dim / max_current
    elif min_current < min_dim:
        scale = min_dim / min_current

    if scale == 1.0:
        return image, 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

# ---------------- CORE ANALYSIS ----------------

def _run_facemesh_single(image_bgr):
    """Internal helper: run FaceMesh once on a BGR image."""
    h, w, _ = image_bgr.shape
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3  # slightly easier threshold
    ) as face_mesh:
        results = face_mesh.process(rgb_image)

    return results, h, w


def analyze_lateral_angles(image_path, selected_angles, progress_callback=None, draw=True):
    """
    - First tries on scaled image
    - If no landmarks, tries again on original image
    - Draws landmarks (white dots) and smaller text for angles
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Error loading image.")

    scaled, scale = limit_image_size_cv2(original, max_dim=1600, min_dim=600)

    if progress_callback:
        progress_callback('start')

    # FIRST ATTEMPT: scaled image
    results, h, w = _run_facemesh_single(scaled)

    # SECOND ATTEMPT: original image if detection failed
    used_image = scaled
    used_scale = scale
    if not results.multi_face_landmarks:
        results, h, w = _run_facemesh_single(original)
        used_image = original
        used_scale = 1.0

    if not results.multi_face_landmarks:
        if progress_callback:
            progress_callback('stop')
        raise ValueError(
            "No face landmarks detected.\n\n"
            "Tips:\n"
            "- Use a clear lateral profile (only one side of the face visible)\n"
            "- Good lighting, no heavy shadows\n"
            "- Face should occupy at least 50–70% of the image height\n"
            "- Avoid very blurry or overexposed photos"
        )

    image = used_image
    h, w, _ = image.shape

    landmarks = [(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark]
    pts = lambda idx: tuple(map(int, landmarks[idx]))

    pts_map = {
        'forehead_glabella': pts(10),
        'nose_bridge':       pts(168),
        'nose_tip':          pts(1),
        'nasal_root':        pts(376),
        'chin_tip':          pts(152),
    }

    raw = {
        'Nasofrontal Angle': calculate_angle(
            pts_map['forehead_glabella'],
            pts_map['nose_bridge'],
            pts_map['nose_tip']
        ),
        'Nasolabial Angle': calculate_angle(
            pts_map['nose_tip'],
            pts_map['nasal_root'],
            pts_map['chin_tip']
        ),
        'Nasomental Angle': calculate_angle(
            pts_map['nose_bridge'],
            pts_map['nose_tip'],
            pts_map['chin_tip']
        ),
        'Nasofacial Angle': None
    }

    nf_angle = angle_between_segments(
        pts_map['nasal_root'], pts_map['nose_tip'],
        pts_map['forehead_glabella'], pts_map['chin_tip']
    )
    raw['Nasofacial Angle'] = nf_angle

    if draw:
        # 1) Draw ALL face landmarks as small white dots
        for (x, y) in landmarks:
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)

        # 2) Draw selected angle constructions
        for name, val in raw.items():
            if name not in selected_angles or val is None:
                continue
            p = pts_map
            if name == 'Nasofrontal Angle':
                cv2.line(image, p['forehead_glabella'], p['nose_bridge'], (0, 255, 0), 2)
                cv2.line(image, p['nose_bridge'], p['nose_tip'], (0, 255, 0), 2)
            elif name == 'Nasolabial Angle':
                cv2.line(image, p['nose_tip'], p['nasal_root'], (0, 255, 0), 2)
                cv2.line(image, p['nasal_root'], p['chin_tip'], (0, 255, 0), 2)
            elif name == 'Nasomental Angle':
                cv2.line(image, p['nose_tip'], p['chin_tip'], (0, 255, 0), 2)
            elif name == 'Nasofacial Angle':
                cv2.line(image, p['forehead_glabella'], p['chin_tip'], (255, 0, 0), 2)
                cv2.line(image, p['nasal_root'], p['nose_tip'], (255, 0, 0), 2)

        # 3) Draw angle text ~50% smaller (scale 0.35 instead of 0.7)
        y_text = 30
        for name, val in raw.items():
            if name not in selected_angles or val is None:
                continue
            cv2.putText(
                image,
                f"{name}: {val:.2f} degrees",
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,                 # smaller font size
                (255, 255, 255),
                1,                    # thinner line
                cv2.LINE_AA
            )
            y_text += 18

        base, ext = os.path.splitext(image_path)
        out = f"{base}_processed{ext}"
        cv2.imwrite(out, image)
    else:
        out = image_path

    if progress_callback:
        progress_callback('stop')

    return {n: raw[n] for n in selected_angles if raw[n] is not None}, out

# ---------------- GUI APPLICATION ----------------

class FaceAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Profiloplasty Angle Analysis")
        self.root.geometry("1000x750")

        # --- Patient info frame ---
        info_frame = tk.Frame(root)
        info_frame.pack(pady=5, fill='x', padx=20)

        tk.Label(info_frame, text="Patient Name:").grid(row=0, column=0, sticky='e', padx=5)
        tk.Label(info_frame, text="Patient ID:").grid(row=0, column=2, sticky='e', padx=5)
        tk.Label(info_frame, text="Notes:").grid(row=1, column=0, sticky='ne', padx=5)

        self.entry_name = tk.Entry(info_frame, width=25)
        self.entry_id = tk.Entry(info_frame, width=20)
        self.txt_notes = tk.Text(info_frame, width=40, height=3)

        self.entry_name.grid(row=0, column=1, sticky='w')
        self.entry_id.grid(row=0, column=3, sticky='w')
        self.txt_notes.grid(row=1, column=1, columnspan=3, sticky='we', pady=5)

        # --- Title ---
        tk.Label(root, text="Profiloplasty Face Analysis", font=("Arial", 18, 'bold')).pack(pady=5)

        # --- Controls ---
        ctrl = tk.Frame(root)
        ctrl.pack(pady=5)

        self.upload_btn = tk.Button(ctrl, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=5)

        self.angle_vars = {}
        for i, name in enumerate(NORMS.keys(), start=1):
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(
                ctrl,
                text=name,
                variable=var
            ).grid(row=0, column=i, padx=5)
            self.angle_vars[name] = var

        # --- Progress bar ---
        self.progress = ttk.Progressbar(root, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=10)

        # --- Main frame: image + text ---
        self.frame = tk.Frame(root)
        self.frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.img_label = tk.Label(self.frame, bg="#333333")
        self.img_label.pack(side='left', padx=10, pady=5, expand=True, fill='both')

        self.txt = tk.Text(self.frame, width=50, state='disabled', font=("Arial", 11))
        self.txt.pack(side='right', padx=10, pady=5, fill='both')

        # --- PDF button ---
        self.pdf_btn = tk.Button(
            root,
            text="Generate PDF Report",
            command=self.generate_pdf,
            state='disabled'
        )
        self.pdf_btn.pack(pady=5)

        self.current_angles = None
        self.processed_image_path = None
        self.original_image_path = None

    # -------- Progress callback --------

    def progress_callback(self, status):
        if status == 'start':
            self.progress.start()
            self.root.config(cursor="watch")
            self.upload_btn.config(state='disabled')
            self.pdf_btn.config(state='disabled')
        else:
            self.progress.stop()
            self.root.config(cursor="")
            self.upload_btn.config(state='normal')
            if self.current_angles:
                self.pdf_btn.config(state='normal')

    # -------- Image upload / processing --------

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return

        try:
            pil_img = Image.open(path)
            pil_img = auto_rotate_pil_image(pil_img)
            temp_path = os.path.join(
                SCRIPT_DIR,
                "_temp_profile_image_for_analysis.jpg"
            )
            pil_img.save(temp_path)
            used_path = temp_path
        except Exception:
            used_path = path

        selected = [n for n, v in self.angle_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("No Angles Selected", "Please select at least one angle.")
            return

        try:
            angles, out = analyze_lateral_angles(
                used_path,
                selected,
                self.progress_callback,
                draw=True
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.original_image_path = path
        self.processed_image_path = out
        self.current_angles = angles
        self.display_results()
        self.pdf_btn.config(state='normal')

    # -------- Display results in GUI --------

    def display_results(self):
        if not self.processed_image_path:
            return
        img = Image.open(self.processed_image_path)

        label_width = self.img_label.winfo_width() or 450
        label_height = self.img_label.winfo_height() or 450
        max_w, max_h = label_width, label_height

        w, h = img.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

        tkimg = ImageTk.PhotoImage(img)
        self.img_label.config(image=tkimg)
        self.img_label.image = tkimg

        self.txt.config(state='normal')
        self.txt.delete('1.0', tk.END)

        patient_name = self.entry_name.get().strip()
        patient_id = self.entry_id.get().strip()

        if patient_name or patient_id:
            self.txt.insert(tk.END, "Patient Info\n", ("header",))
            if patient_name:
                self.txt.insert(tk.END, f"  Name: {patient_name}\n")
            if patient_id:
                self.txt.insert(tk.END, f"  ID: {patient_id}\n")
            self.txt.insert(tk.END, "\n")

        self.txt.insert(tk.END, "Angle Analysis\n", ("header",))

        for name, val in self.current_angles.items():
            mean, sd = NORMS.get(name, (None, None))
            interp = INTERPRETATIONS.get(name, "")
            self.txt.insert(tk.END, f"{name}: {val:.2f}°\n")
            if mean is not None and sd is not None:
                normal_range = f"{mean:.1f}° ± {sd:.1f}°"
                status = classify_deviation(val, mean, sd)
                self.txt.insert(tk.END, f"  Normal: {normal_range}\n")
                self.txt.insert(tk.END, f"  Status: {status}\n")
            if interp:
                self.txt.insert(tk.END, f"  Note: {interp}\n")
            self.txt.insert(tk.END, "\n")

        self.txt.tag_configure("header", font=("Arial", 12, "bold"))
        self.txt.config(state='disabled')

    # -------- PDF generation --------

    def generate_pdf(self):
        if not self.current_angles or not self.processed_image_path:
            return

        base_name = os.path.splitext(os.path.basename(self.processed_image_path))[0]
        default_report_name = base_name + "_report.pdf"

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=default_report_name,
            filetypes=[("PDF files", "*.pdf")]
        )
        if not save_path:
            return

        c = canvas.Canvas(save_path)
        width, height = c._pagesize
        margin = 50

        c.setFont(PDF_FONT, 16)
        c.drawCentredString(width / 2, height - margin, "Profiloplasty Angle Analysis Report")

        c.setFont(PDF_FONT, 11)
        y_header = height - margin - 30

        patient_name = self.entry_name.get().strip()
        patient_id = self.entry_id.get().strip()
        notes = self.txt_notes.get("1.0", tk.END).strip()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if patient_name:
            c.drawString(margin, y_header, f"Name: {patient_name}")
            y_header -= 14
        if patient_id:
            c.drawString(margin, y_header, f"ID: {patient_id}")
            y_header -= 14
        c.drawString(margin, y_header, f"Date/Time: {timestamp}")
        y_header -= 20
        if notes:
            c.drawString(margin, y_header, "Clinical Notes:")
            y_header -= 14
            max_width = width - 2 * margin
            lines = []
            words = notes.split()
            line = ""
            for w_ in words:
                trial = (line + " " + w_).strip()
                if c.stringWidth(trial, PDF_FONT, 11) > max_width:
                    lines.append(line)
                    line = w_
                else:
                    line = trial
            if line:
                lines.append(line)
            for ln in lines:
                c.drawString(margin, y_header, ln)
                y_header -= 14

        img_w, img_h = 300, 300
        x_img = (width - img_w) / 2
        y_img = y_header - img_h - 20
        if y_img < 150:
            y_img = 150
        c.drawImage(
            self.processed_image_path,
            x_img, y_img,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            anchor='c'
        )

        y_text = y_img - 30
        c.setFont(PDF_FONT, 12)
        for name, val in self.current_angles.items():
            mean, sd = NORMS.get(name, (None, None))
            interp = INTERPRETATIONS.get(name, "")
            c.drawString(margin, y_text, f"{name}: {val:.2f}°")
            y_text -= 16
            if mean is not None and sd is not None:
                normal_range = f"{mean:.1f}° ± {sd:.1f}°"
                status = classify_deviation(val, mean, sd)
                c.drawString(margin, y_text, f"Normal: {normal_range}")
                y_text -= 16
                c.drawString(margin, y_text, f"Status: {status}")
                y_text -= 16
            if interp:
                c.drawString(margin, y_text, f"Interpretation: {interp}")
                y_text -= 24

            if y_text < 100:
                c.showPage()
                c.setFont(PDF_FONT, 12)
                y_text = height - margin

        c.setFont(PDF_FONT, 9)
        c.drawCentredString(
            width / 2,
            40,
            "Generated by PocketDoc Profiloplasty Angle Tool"
        )

        c.save()
        messagebox.showinfo("Saved", f"PDF report saved to:\n{save_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = FaceAnalysisApp(root)
    root.mainloop()
