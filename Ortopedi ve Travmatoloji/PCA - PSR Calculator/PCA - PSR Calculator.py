import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import tempfile

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.utils import ImageReader


# ======================================================
# Geometry Helpers
# ======================================================

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
          (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
          (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    def within(a, b, c):
        return min(a, b) - 1 <= c <= max(a, b) + 1

    if (within(x1, x2, px) and within(y1, y2, py) and
            within(x3, x4, px) and within(y3, y4, py)):
        return (px, py)
    return None


def line_angle_degrees(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    return angle


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# ======================================================
# PSR + PCA Computation
# ======================================================

def compute_psr_pca(image, frac_p1, frac_p2):
    """
    image   : BGR OpenCV image
    frac_p1, frac_p2 : fracture line endpoints in IMAGE coordinates
    Returns:
        PSR, PCA, candidate_lines, intersections, WH, PS, best_angle_pair, ps_pair
        ps_pair = (pt_left, pt_right) intersection pair giving max PS
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    raw_lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=70, minLineLength=40, maxLineGap=10
    )

    if raw_lines is None:
        return None, None, [], [], None, None, None, None

    frac_p1 = tuple(map(float, frac_p1))
    frac_p2 = tuple(map(float, frac_p2))

    WH = distance(frac_p1, frac_p2)
    if WH < 1:
        return None, None, [], [], None, None, None, None

    candidate_lines = []
    intersections = []

    for l in raw_lines:
        x1, y1, x2, y2 = l[0]
        p1, p2 = (float(x1), float(y1)), (float(x2), float(y2))
        ang = line_angle_degrees(p1, p2)

        # ignore almost horizontal anatomy
        if 10 <= ang <= 170:
            ip = line_intersection(p1, p2, frac_p1, frac_p2)
            if ip is not None:
                candidate_lines.append((p1, p2, ang))
                intersections.append(ip)

    if len(intersections) < 2:
        return None, None, candidate_lines, intersections, WH, None, None, None

    # --- PSR ---
    max_ps = 0.0
    ps_pair = None
    for i in range(len(intersections)):
        for j in range(i + 1, len(intersections)):
            d = distance(intersections[i], intersections[j])
            if d > max_ps:
                max_ps = d
                ps_pair = (intersections[i], intersections[j])

    PS = max_ps
    PSR = PS / WH

    # --- PCA ---
    max_angle = 0.0
    best_pair = None
    angles = [cl[2] for cl in candidate_lines]
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            diff = abs(angles[i] - angles[j])
            diff = min(diff, 180 - diff)
            if diff > max_angle:
                max_angle = diff
                best_pair = (i, j)

    PCA = max_angle

    return PSR, PCA, candidate_lines, intersections, WH, PS, best_pair, ps_pair


# ======================================================
# Tkinter GUI Application
# ======================================================

class PSR_PCA_App:
    def __init__(self, root):
        self.root = root
        root.title("PSR / PCA Measurement Tool")
        root.geometry("1100x800")

        self.cv_img = None
        self.fracture_points = []    # in canvas coords
        self.cursor_pos = None       # for live line preview

        # For PDF
        self.last_annotated = None
        self.last_PSR = None
        self.last_PCA = None
        self.last_WH = None
        self.last_PS = None

        # --- Layout ---
        self.canvas = tk.Canvas(root, width=700, height=700, bg='black')
        self.canvas.pack(side="left", padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_move)

        right_frame = tk.Frame(root)
        right_frame.pack(side="right", fill="y", padx=10, pady=10)

        tk.Button(right_frame, text="Load X-ray", width=20,
                  command=self.load_image).pack(pady=10)
        tk.Button(right_frame, text="Compute PSR / PCA",
                  command=self.compute).pack(pady=10)
        tk.Button(right_frame, text="Export PDF Report",
                  command=self.export_pdf).pack(pady=10)

        self.psr_label = tk.Label(right_frame, text="PSR: -", font=("Arial", 14))
        self.psr_label.pack(pady=10)

        self.pca_label = tk.Label(right_frame, text="PCA: -", font=("Arial", 14))
        self.pca_label.pack(pady=10)

        self.status_label = tk.Label(
            right_frame,
            text="Load image, then click 2 points on cortex at fracture level",
            fg="blue", wraplength=250, justify="left"
        )
        self.status_label.pack(pady=10)

        self.tk_image = None

    # ------------- Display helpers -------------

    def display_array(self, arr_bgr):
        rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

    def show_base_image(self):
        if self.cv_img is None:
            return
        base = cv2.resize(self.cv_img, (700, 700))
        self.display_array(base)

    # ------------- Events -------------

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot load image.")
            return

        self.cv_img = img
        self.fracture_points.clear()
        self.cursor_pos = None

        self.last_annotated = None
        self.last_PSR = None
        self.last_PCA = None
        self.last_WH = None
        self.last_PS = None

        self.psr_label.config(text="PSR: -")
        self.pca_label.config(text="PCA: -")
        self.status_label.config(
            text="Click 2 points on the cortex at fracture level"
        )

        self.show_base_image()

    def on_click(self, event):
        if self.cv_img is None:
            return

        x = max(0, min(699, event.x))
        y = max(0, min(699, event.y))

        if len(self.fracture_points) < 2:
            self.fracture_points.append((x, y))
            if len(self.fracture_points) == 2:
                self.status_label.config(text="Press 'Compute PSR / PCA'")
        else:
            # third click → reset
            self.fracture_points = [(x, y)]
            self.status_label.config(text="Select second point for fracture line")

        self.update_preview()

    def on_move(self, event):
        if self.cv_img is None:
            return
        if len(self.fracture_points) == 1:
            x = max(0, min(699, event.x))
            y = max(0, min(699, event.y))
            self.cursor_pos = (x, y)
            self.update_preview()
        else:
            self.cursor_pos = None

    # ------------- Preview drawing -------------

    def update_preview(self):
        if self.cv_img is None:
            return

        base = cv2.resize(self.cv_img, (700, 700))

        # small white dots for your manual fracture endpoints
        for p in self.fracture_points:
            cv2.circle(base, p, 3, (255, 255, 255), -1)

        # line preview (black)
        if len(self.fracture_points) == 2:
            cv2.line(base, self.fracture_points[0],
                     self.fracture_points[1], (0, 0, 0), 2)
        elif len(self.fracture_points) == 1 and self.cursor_pos is not None:
            cv2.line(base, self.fracture_points[0],
                     self.cursor_pos, (0, 0, 0), 1)

        self.display_array(base)

    # ------------- Core computation -------------

    def compute(self):
        if self.cv_img is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        if len(self.fracture_points) != 2:
            messagebox.showwarning(
                "Warning", "Select EXACTLY 2 points on the fracture level."
            )
            return

        h, w = self.cv_img.shape[:2]
        scale_x = w / 700.0
        scale_y = h / 700.0

        p1_canvas, p2_canvas = self.fracture_points
        p1 = (p1_canvas[0] * scale_x, p1_canvas[1] * scale_y)
        p2 = (p2_canvas[0] * scale_x, p2_canvas[1] * scale_y)

        PSR, PCA, lines, intersections, WH, PS, best_pair, ps_pair = compute_psr_pca(
            self.cv_img, p1, p2
        )

        if PSR is None or PCA is None:
            messagebox.showerror(
                "Error",
                "Could not detect enough K-wires intersecting the fracture line."
            )
            return

        self.psr_label.config(text=f"PSR: {PSR:.3f}")
        self.pca_label.config(text=f"PCA: {PCA:.1f} degree")

        # ----- Annotated image -----
        annotated = self.cv_img.copy()

        # fracture line (WH) – black
        cv2.line(
            annotated,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (0, 0, 0), 2
        )

        # K-wires (blue), highlight best-angle pair thicker
        highlight = set(best_pair) if best_pair is not None else set()
        for idx, (a, b, ang) in enumerate(lines):
            thickness = 3 if idx in highlight else 2
            cv2.line(
                annotated,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                (255, 0, 0),  # blue
                thickness
            )

        # PS line (green) + only two outermost dots (white)
        if ps_pair is not None:
            pt1, pt2 = ps_pair
            p1i = (int(pt1[0]), int(pt1[1]))
            p2i = (int(pt2[0]), int(pt2[1]))
            # green PS segment
            cv2.line(annotated, p1i, p2i, (0, 255, 0), 2)
            # small white dots only at these two points
            cv2.circle(annotated, p1i, 4, (255, 255, 255), -1)
            cv2.circle(annotated, p2i, 4, (255, 255, 255), -1)

        # ----- Text with semi-transparent box (top-left) -----
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thick_main = 1

        lines_text = [
            f"PSR = {PSR:.3f}",
            f"PS/WH = {PS:.1f} / {WH:.1f}"
            if WH is not None and PS is not None else "",
            f"PCA = {PCA:.1f} degree"
        ]
        lines_text = [t for t in lines_text if t]

        padding_x = 8
        padding_y = 6
        line_spacing = 4

        widths, heights = [], []
        for t in lines_text:
            (w_txt, h_txt), baseline = cv2.getTextSize(
                t, font, font_scale, thick_main
            )
            widths.append(w_txt)
            heights.append(h_txt)

        box_w = max(widths) + 2 * padding_x
        box_h = sum(heights) + (len(lines_text) - 1) * line_spacing + 2 * padding_y

        box_x1, box_y1 = 10, 10
        box_x2, box_y2 = box_x1 + box_w, box_y1 + box_h

        overlay = annotated.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        alpha = 0.4
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

        y_cursor = box_y1 + padding_y
        for idx, t in enumerate(lines_text):
            h_txt = heights[idx]
            cv2.putText(
                annotated, t, (box_x1 + padding_x, y_cursor + h_txt),
                font, font_scale, (0, 0, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                annotated, t, (box_x1 + padding_x, y_cursor + h_txt),
                font, font_scale, (255, 255, 255), thick_main, cv2.LINE_AA
            )
            y_cursor += h_txt + line_spacing

        # Save for PDF
        self.last_annotated = annotated
        self.last_PSR = PSR
        self.last_PCA = PCA
        self.last_WH = WH
        self.last_PS = PS

        disp = cv2.resize(annotated, (700, 700))
        self.display_array(disp)

        self.status_label.config(
            text="Computation done. You can now export the PDF report."
        )

    # ------------- PDF Export -------------

    def export_pdf(self):
        if self.cv_img is None or self.last_annotated is None:
            messagebox.showwarning(
                "Warning",
                "You need to load an image and run 'Compute PSR / PCA' first."
            )
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save PDF report"
        )
        if not save_path:
            return

        raw_fd, raw_path = tempfile.mkstemp(suffix=".png")
        ann_fd, ann_path = tempfile.mkstemp(suffix=".png")
        os.close(raw_fd)
        os.close(ann_fd)

        try:
            cv2.imwrite(raw_path, self.cv_img)
            cv2.imwrite(ann_path, self.last_annotated)

            c = pdfcanvas.Canvas(save_path, pagesize=A4)
            page_w, page_h = A4

            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, page_h - 40,
                         "Supracondylar Humerus – K-wire Construct Report")

            img_raw = ImageReader(raw_path)
            img_ann = ImageReader(ann_path)
            iw_raw, ih_raw = img_raw.getSize()
            iw_ann, ih_ann = img_ann.getSize()

            max_w = (page_w - 100) / 2.0
            scale_raw = max_w / iw_raw
            scale_ann = max_w / iw_ann
            h_raw = ih_raw * scale_raw
            h_ann = ih_ann * scale_ann
            img_h = min(h_raw, h_ann)

            img_y = page_h - 80 - img_h
            c.drawImage(img_raw, 40,
                        img_y,
                        width=max_w, height=img_h,
                        preserveAspectRatio=True)
            c.drawImage(img_ann, 40 + max_w + 20,
                        img_y,
                        width=max_w, height=img_h,
                        preserveAspectRatio=True)

            # Text block
            c.setFont("Helvetica", 10)
            text_start_y = img_y - 30
            text = c.beginText(40, text_start_y)

            psr = self.last_PSR
            pca = self.last_PCA

            text.textLine(f"Measured PSR (Pin Separation Ratio): {psr:.3f}")
            if self.last_WH is not None and self.last_PS is not None:
                text.textLine(
                    f"  PSR = PS / WH = {self.last_PS:.1f} / {self.last_WH:.1f}"
                )
            text.textLine(
                "  PS = distance between the two outermost K-wire intersections "
                "with the fracture line."
            )
            text.textLine(
                "  WH = humeral width along the user-defined fracture-level line."
            )
            text.textLine("")

            text.textLine(
                f"Measured PCA (Pin Crossing Angle): {pca:.1f} degrees"
            )
            text.textLine(
                "  PCA = largest angle between any two K-wires "
                "in the coronal plane."
            )
            text.textLine("")

            text.textLine("Clinical interpretation (based on current literature):")

            if psr >= 0.33:
                text.textLine(
                    "- PSR ≥ 0.33 (≥ one-third of humeral width):"
                )
                text.textLine(
                    "  pin spread is generally considered adequate for coronal stability."
                )
            else:
                text.textLine("- PSR < 0.33:")
                text.textLine(
                    "  pin spread is relatively narrow and may be associated with "
                    "increased risk of loss of reduction,"
                )
                text.textLine(
                    "  especially in the presence of medial comminution or "
                    "only-lateral pin constructs."
                )

            if pca >= 90:
                text.textLine("- PCA around 90 degrees or higher:")
                text.textLine(
                    "  crossing angle is high and biomechanically favorable for "
                    "torsional stability."
                )
            elif pca >= 70:
                text.textLine("- PCA between 70 and 90 degrees:")
                text.textLine(
                    "  crossing angle is moderate and usually acceptable,"
                )
                text.textLine("  but not maximal.")
            else:
                text.textLine("- PCA below 70 degrees:")
                text.textLine(
                    "  crossing angle is relatively low; wires are more parallel,"
                )
                text.textLine(
                    "  which can reduce torsional stiffness."
                )

            text.textLine("")
            text.textLine(
                "Note: This tool provides geometric measurements only and does not "
                "replace clinical judgment, fracture pattern "
            )
            text.textLine(
                "assessment, or intra-operative decision-making."
            )

            c.drawText(text)
            c.showPage()
            c.save()

            messagebox.showinfo("Done", f"PDF report saved to:\n{save_path}")

        finally:
            try:
                os.remove(raw_path)
            except OSError:
                pass
            try:
                os.remove(ann_path)
            except OSError:
                pass


# ======================================================
# Run App
# ======================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PSR_PCA_App(root)
    root.mainloop()
