import io
import os
import json
from typing import List, Dict, Any, Tuple, Optional

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# YOLOv8 (segmentation-capable)
from ultralytics import YOLO


# -------------------------
# Configuration
# -------------------------
APP_TITLE = "Ulna/Radius Segmentation + Fracture Reporter (Reports Only)"

# Your model path
MODEL_PATH = "/Users/avnitan/PycharmProjects/TestProject/PocketDoc/Modeller/UlnaFracture.pt"

# Class mapping (as provided):
# 0: fracture, 1: nondisplaced, 2: radius, 4: ulna
CLASS_MAP = {
    0: "fracture",
    1: "nondisplaced",
    2: "radius",
    4: "ulna",
}

FRACTURE_ID = 0
NOND_ID = 1
RADIUS_ID = 2
ULNA_ID = 4

# Report figures
FIG_NONDISPLACED = "/Users/avnitan/Desktop/nondisplaced.png"
FIG_FRACTURE = "/Users/avnitan/Desktop/fracture.png"

# Visual styles
BOX_COLORS = {
    FRACTURE_ID: (255, 64, 64),      # red-ish
    NOND_ID: (255, 160, 64),         # orange-ish
}
MASK_COLORS = {
    RADIUS_ID: (64, 128, 255, 90),   # semi-transparent blue (RGBA)
    ULNA_ID: (64, 255, 128, 90),     # semi-transparent green (RGBA)
}
LABEL_BG = (230, 57, 70)  # label red bar for boxes text
LABEL_FG = (255, 255, 255)


# -------------------------
# Model adapter
# -------------------------
def load_model():
    """Load YOLOv8 model (segmentation-capable)."""
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded from: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def run_inference(model, pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Run YOLOv8 inference. Return a list of detections with either masks (for ulna & radius)
    or bounding boxes (for fracture & nondisplaced).

    Output schema for each detection:
      {
        "class_id": int,
        "class_name": str,
        "conf": float,
        "bbox": [x1,y1,x2,y2],            # always present
        "polys": [ [(x,y),...], ... ]     # present ONLY for mask classes (ulna, radius)
      }
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    # imgsz and conf can be tuned to your model
    results = model.predict(pil_image, imgsz=640, conf=0.25, verbose=False)
    detections: List[Dict[str, Any]] = []

    if not results or len(results) == 0:
        return detections

    res = results[0]
    boxes = getattr(res, "boxes", None)
    masks = getattr(res, "masks", None)  # for seg models; None for pure detect models

    if boxes is None:
        return detections

    # masks.xy is a list (len=n) of polygons (each polygon is Nx2 array in pixel coords)
    masks_xy = []
    if masks is not None and getattr(masks, "xy", None) is not None:
        masks_xy = masks.xy  # list of list-of-xy for each instance

    n = len(boxes)
    for i in range(n):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())

        det: Dict[str, Any] = {
            "class_id": cls_id,
            "class_name": CLASS_MAP.get(cls_id, f"class_{cls_id}"),
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
        }

        # If it's ulna or radius and masks exist, attach polygons
        if cls_id in (ULNA_ID, RADIUS_ID) and masks_xy and i < len(masks_xy):
            # masks_xy[i] may be a list of polygons; normalize to list-of-lists
            polyset = masks_xy[i]
            # polyset can be a single Nx2 array or list of such arrays
            polys: List[List[Tuple[float, float]]] = []
            if isinstance(polyset, list):
                for poly in polyset:
                    polys.append([(float(px), float(py)) for px, py in poly.tolist()])
            else:
                polys.append([(float(px), float(py)) for px, py in polyset.tolist()])
            det["polys"] = polys

        detections.append(det)

    print("Detected classes:", [d["class_name"] for d in detections])
    return detections


# -------------------------
# Decision and report logic
# -------------------------
def decide_report(dets: List[Dict[str, Any]]) -> Tuple[str, str, Optional[str]]:
    """
    Return (title, body_text, figure_path_to_embed_or_None).

    Rules (priority: fracture > nondisplaced if both):
      - If ulna AND fracture  -> oblique fracture fixation principles + fracture figure
      - If ulna AND nondisplaced -> transverse fixation principles + nondisplaced figure
    """
    classes = {d["class_id"] for d in dets}
    has_ulna = ULNA_ID in classes
    has_fracture = FRACTURE_ID in classes
    has_nond = NOND_ID in classes

    if has_ulna and has_fracture:
        title = "Oblique Ulna Fracture – Fixation Principles"
        body = (
            "Principle: Interfragmentary compression across the fracture line.\n\n"
            "- Primary fixation: Lag screw perpendicular to fracture plane.\n"
            "- Neutralization: Plate to protect the lag construct from shear/torsion.\n"
            "- Screws: ≥3 bicortical screws per main fragment.\n"
            "- Plate: 3.5 mm LCP/DCP; consider longer plate near metaphysis.\n"
            "- Reduction: Restore ulnar axis on orthogonal views; avoid gapping.\n"
            "- Pitfalls: Avoid screw too close to fracture edge; beware occult comminution.\n"
            "\nNote: Educational decision-support only; final surgical plan is surgeon’s discretion."
        )
        figure = FIG_FRACTURE if os.path.exists(FIG_FRACTURE) else None
        return title, body, figure

    if has_ulna and has_nond:
        title = "Transverse (Nondisplaced) Ulna Fracture – Fixation Principles"
        body = (
            "Principle: Axial compression across a transverse fracture for primary bone healing.\n\n"
            "- Primary fixation: Compression plating (eccentric drilling or device).\n"
            "- Plate: 3.5 mm compression plate (LCP/DCP) spanning fracture.\n"
            "- Screws: ≥3 bicortical screws each side, symmetric spread.\n"
            "- Reduction: Maintain length, alignment, rotation; verify on PA and lateral views.\n"
            "- Strategy: Prefer compression for clean transverse; bridge if occult comminution.\n"
            "- Pitfalls: Inadequate compression or too few screws → loss of fixation risk.\n"
            "\nNote: Educational decision-support only; final surgical plan is surgeon’s discretion."
        )
        figure = FIG_NONDISPLACED if os.path.exists(FIG_NONDISPLACED) else None
        return title, body, figure

    title = "No Ulna Fracture Principle Triggered"
    body = (
        "Rules not satisfied:\n"
        "- Needs {ulna + fracture} for oblique principles, OR\n"
        "- Needs {ulna + nondisplaced} for transverse principles.\n\n"
        "If you believe this is an ulna fracture, verify model outputs or try a clearer image."
    )
    return title, body, None


def export_pdf(path: str, title: str, body: str, overlay_img: Image.Image, figure_path: Optional[str] = None):
    """Save report as PDF with detection overlay (first) and optional principle figure (second)."""
    # Prepare overlay image
    buf_overlay = io.BytesIO()
    overlay_img.save(buf_overlay, format="PNG")
    buf_overlay.seek(0)
    img_overlay_reader = ImageReader(buf_overlay)

    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, H - 2.3 * cm, title)

    # Place overlay image
    max_w = W - 4 * cm
    iw, ih = overlay_img.size
    scale = min(max_w / iw, 10 * cm / ih)  # a bit smaller to leave room for figure & text
    nw, nh = iw * scale, ih * scale
    y_cursor = H - 2.3 * cm - nh - 0.3 * cm
    c.drawImage(img_overlay_reader, 2 * cm, y_cursor, width=nw, height=nh)

    # Optional principle figure
    if figure_path and os.path.exists(figure_path):
        try:
            with Image.open(figure_path) as fig_im:
                fig_buf = io.BytesIO()
                fig_im.save(fig_buf, format="PNG")
                fig_buf.seek(0)
                img_fig_reader = ImageReader(fig_buf)

                # reserve up to ~8 cm height for figure
                max_fig_h = 8 * cm
                fw, fh = fig_im.size
                fscale = min(max_w / fw, max_fig_h / fh)
                fnw, fnh = fw * fscale, fh * fscale

                y_cursor = y_cursor - fnh - 0.5 * cm
                if y_cursor < 5 * cm:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(2 * cm, H - 2.3 * cm, title + " (cont.)")
                    y_cursor = H - 2.8 * cm

                c.drawImage(img_fig_reader, 2 * cm, y_cursor, width=fnw, height=fnh)

        except Exception as e:
            print(f"Warning: Failed to embed principle figure: {e}")

    # Body text
    y_text_top = y_cursor - 0.8 * cm
    if y_text_top < 3 * cm:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, H - 2.3 * cm, title + " (cont.)")
        y_text_top = H - 3.0 * cm

    c.setFont("Helvetica", 11)
    text = c.beginText(2 * cm, y_text_top)

    def wrap(s, width=95):
        lines = []
        for p in s.split("\n"):
            words, line = p.split(), ""
            if not words:
                lines.append("")
                continue
            for w in words:
                if len(line) + len(w) + 1 <= width:
                    line = (line + " " + w).strip()
                else:
                    lines.append(line)
                    line = w
            lines.append(line)
        return lines

    for ln in wrap(body):
        if text.getY() < 2.0 * cm:
            c.drawText(text)
            c.showPage()
            c.setFont("Helvetica", 11)
            text = c.beginText(2 * cm, H - 3.0 * cm)
        text.textLine(ln)

    c.drawText(text)
    c.showPage()
    c.save()


# -------------------------
# Rendering helpers
# -------------------------
def draw_segmentation_and_boxes(pil_img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """
    Draw semi-transparent masks for ulna & radius; draw boxes for fracture & nondisplaced.
    """
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw_box = ImageDraw.Draw(base)  # boxes directly on base (for crisp text bar)

    # First: fill masks (ulna/radius)
    for det in detections:
        cid = det["class_id"]
        polys = det.get("polys")
        if polys and cid in MASK_COLORS:
            fill = MASK_COLORS[cid]
            for poly in polys:
                # PIL expects tuples
                pts = [(float(x), float(y)) for x, y in poly]
                # sometimes masks can be small/degenerate; guard
                if len(pts) >= 3:
                    draw_ov.polygon(pts, fill=fill)

    # Composite masks over base
    masked = Image.alpha_composite(base, overlay)

    # Then: draw bounding boxes for fracture/nondisplaced
    draw_box = ImageDraw.Draw(masked)
    for det in detections:
        cid = det["class_id"]
        if cid in (FRACTURE_ID, NOND_ID):
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = BOX_COLORS.get(cid, (255, 0, 0))
            draw_box.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            label = f'{det["class_name"]} {det["conf"]:.2f}'
            tw = 7 * len(label) + 6
            y_top = max(0, y1 - 18)
            draw_box.rectangle([(x1, y_top), (x1 + tw, y_top + 18)], fill=LABEL_BG)
            draw_box.text((x1 + 3, y_top + 3), label, fill=LABEL_FG)

    return masked.convert("RGB")


# -------------------------
# Tkinter UI
# -------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.model = load_model()

        self.orig_image: Optional[Image.Image] = None
        self.overlay_img: Optional[Image.Image] = None
        self.detections: List[Dict[str, Any]] = []

        self.report_title = ""
        self.report_body = ""
        self.report_figure_path: Optional[str] = None

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Button(top, text="Open Image", width=15, command=self.open_image).pack(side=tk.LEFT)
        tk.Button(top, text="Run Detection", width=15, command=self.run_detection).pack(side=tk.LEFT, padx=6)
        tk.Button(top, text="Export PDF", width=15, command=self.export_report).pack(side=tk.LEFT, padx=6)
        tk.Button(top, text="Save Detections (JSON)", width=20, command=self.save_json).pack(side=tk.LEFT, padx=6)

        mid = tk.Frame(self.root)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(mid, width=720, height=720, bg="#1c1c1c", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 0))

        tk.Label(right, text="Generated Report", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.txt_report = tk.Text(right, width=52, height=28, wrap="word")
        self.txt_report.pack(fill=tk.BOTH, expand=True)

        tk.Label(right, text="Figure", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(8, 2))
        self.figure_label = tk.Label(right)
        self.figure_label.pack(anchor="w")

        self.lbl_dets = tk.Label(self.root, text="Detections: —")
        self.lbl_dets.pack(anchor="w", padx=10, pady=(0, 6))

        # Redraw on resize
        self.canvas.bind("<Configure>", lambda e: self._redraw_canvas())

    # --- UI actions ---
    def open_image(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select X-ray image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*")
            ],
        )
        if not path:
            return
        try:
            self.orig_image = Image.open(path).convert("RGB")
            self.overlay_img = None
            self.detections = []
            self._show_on_canvas(self.orig_image)
            self.txt_report.delete("1.0", tk.END)
            self.figure_label.config(image="", text="")
            self.lbl_dets.config(text="Detections: —")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def run_detection(self):
        if self.orig_image is None:
            messagebox.showinfo("Info", "Please open an image first.")
            return
        try:
            self.detections = run_inference(self.model, self.orig_image)
            self.overlay_img = draw_segmentation_and_boxes(self.orig_image, self.detections)
            self._show_on_canvas(self.overlay_img)

            self.report_title, self.report_body, self.report_figure_path = decide_report(self.detections)
            self.txt_report.delete("1.0", tk.END)
            self.txt_report.insert(tk.END, f"{self.report_title}\n\n{self.report_body}")
            self._update_det_label()
            self._update_report_figure()
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{e}")

    def export_report(self):
        if not self.overlay_img:
            messagebox.showinfo("Info", "Run detection first.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All files", "*")],
            title="Save Report"
        )
        if not path:
            return
        try:
            export_pdf(
                path,
                self.report_title or "Ulna Fracture Report",
                self.report_body or "",
                self.overlay_img,
                figure_path=self.report_figure_path
            )
            messagebox.showinfo("Saved", f"Report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF:\n{e}")

    def save_json(self):
        if not self.detections:
            messagebox.showinfo("Info", "No detections to save.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
            title="Save Detections"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.detections, f, indent=2)
            messagebox.showinfo("Saved", f"Detections saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON:\n{e}")

    # --- Rendering helpers ---
    def _show_on_canvas(self, pil_img: Image.Image):
        # Fit to canvas while preserving aspect
        c_w = max(1, int(self.canvas.winfo_width() or 720))
        c_h = max(1, int(self.canvas.winfo_height() or 720))
        img_w, img_h = pil_img.size
        scale = min(c_w / img_w, c_h / img_h)
        new_img = pil_img.resize((max(1, int(img_w * scale)), max(1, int(img_h * scale))), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(new_img)  # keep reference
        self.canvas.delete("all")
        self.canvas.create_image(c_w // 2, c_h // 2, image=self._tk_img)

    def _redraw_canvas(self):
        if self.overlay_img is not None:
            self._show_on_canvas(self.overlay_img)
        elif self.orig_image is not None:
            self._show_on_canvas(self.orig_image)

    def _update_det_label(self):
        if not self.detections:
            self.lbl_dets.config(text="Detections: —")
            return
        lines = []
        for d in self.detections:
            xy = [round(float(v), 1) for v in d["bbox"]]
            extra = " (mask)" if "polys" in d else " (box)"
            lines.append(f'{d["class_name"]} (id={d["class_id"]}){extra} | conf={d["conf"]:.2f} | bbox={xy}')
        self.lbl_dets.config(text="Detections:\n" + "\n".join(lines))

    def _update_report_figure(self):
        if not self.report_figure_path or not os.path.exists(self.report_figure_path):
            self.figure_label.config(image="", text="(No figure)")
            return
        try:
            fig = Image.open(self.report_figure_path).convert("RGB")
            # shrink to a reasonable width
            max_w = 360
            scale = min(1.0, max_w / fig.width)
            fig = fig.resize((int(fig.width * scale), int(fig.height * scale)), Image.LANCZOS)
            self._tk_fig = ImageTk.PhotoImage(fig)
            self.figure_label.config(image=self._tk_fig, text="")
        except Exception as e:
            self.figure_label.config(image="", text=f"(Failed to load figure: {e})")


# -------------------------
# Main
# -------------------------
def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
