import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from datetime import datetime
import tempfile
import os


class YOLOMelanomaApp:
    def __init__(
        self,
        window,
        video_source=0,
        model_path=None,
        malign_cls_id=1,
        benign_cls_id=0,
        malign_threshold=0.8,
    ):
        self.window = window
        self.window.title("Melanoma Detection – Experimental Tool")

        self.video_source = video_source
        self.malign_cls_id = malign_cls_id
        self.benign_cls_id = benign_cls_id
        self.malign_threshold = malign_threshold

        self.running = False  # for webcam loop

        # For PDF report (used for BOTH live cam and uploaded images)
        self.last_original_image = None   # PIL.Image
        self.last_annotated_image = None  # PIL.Image
        self.last_prediction_label = None
        self.last_prediction_conf = None

        # --- Model path handling ---
        if model_path is None:
            model_path = Path(
                "/Users/avnitan/PycharmProjects/TestProject/PocketDoc/Modeller/Dermatology/Melanoma/melanom.pt"
            )

        # Load YOLO model
        try:
            self.model = YOLO(str(model_path))
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model:\n{e}")
            self.window.destroy()
            return

        # Canvas (will be resized to video/image size later)
        self.canvas = tk.Canvas(window, width=800, height=600, bg="black")
        self.canvas.pack(padx=10, pady=10)

        # Button frame
        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=5)

        self.btn_load = tk.Button(
            btn_frame,
            text="Load Clinical Image",
            command=self.load_image,
            width=20
        )
        self.btn_load.grid(row=0, column=0, padx=5)

        self.btn_report = tk.Button(
            btn_frame,
            text="Generate PDF Report",
            command=self.generate_pdf_report,
            width=20
        )
        self.btn_report.grid(row=0, column=1, padx=5)

        self.btn_quit = tk.Button(
            btn_frame,
            text="Quit",
            command=self.on_closing,
            width=10
        )
        self.btn_quit.grid(row=0, column=2, padx=5)

        # Caution label – explicit about *all* outcomes
        self.caution_label = tk.Label(
            window,
            text=(
                "This experimental tool does NOT replace a dermatologist.\n"
                "Regardless of the AI result (benign, malignant or doubtful), "
                "any suspicious lesion must be evaluated by a specialist."
            ),
            font=("Helvetica", 11),
            fg="red",
        )
        self.caution_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # --- Webcam: start live cam at beginning ---
        self.vid = None
        if self.video_source is not None:
            self.vid = cv2.VideoCapture(self.video_source)
            if not self.vid.isOpened():
                messagebox.showerror(
                    "Video Error",
                    "Unable to open webcam. You can still use image upload."
                )
                self.vid = None
            else:
                self.running = True
                self.update_video()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ---------- CORE INFERENCE LOGIC WITH YOUR RULES ----------

    def run_inference_on_frame(self, frame_bgr):
        """
        Takes a BGR frame, runs YOLO, and returns:
        - annotated_frame_bgr
        - prediction_label: "malign", "benign", "doubtful", "none", or "error"
        - prediction_conf: float or None (uses malignant/benign confidence when doubtful)
        """
        try:
            results = self.model(frame_bgr, verbose=False)
            if not results:
                boxes = None
            else:
                boxes = results[0].boxes

            best_malign_conf = 0.0
            best_benign_conf = 0.0
            best_malign_box = None
            best_benign_box = None

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    conf = float(box.conf.item())
                    cls = int(box.cls.item())
                    if cls == self.malign_cls_id:
                        if conf > best_malign_conf:
                            best_malign_conf = conf
                            best_malign_box = box.xyxy.cpu().numpy()[0]
                    elif cls == self.benign_cls_id:
                        if conf > best_benign_conf:
                            best_benign_conf = conf
                            best_benign_box = box.xyxy.cpu().numpy()[0]

            annotated = frame_bgr.copy()

            label = "none"
            conf_to_show = None
            box_to_draw = None

            # ---- Your rules ----
            # 1) Both benign and malign with near-same percentages → Doubtful
            #    Treat “near” as within ±10% (0.1) and malign >= 0.3
            if best_malign_conf >= 0.3 and best_benign_conf > 0 and abs(best_malign_conf - best_benign_conf) <= 0.1:
                label = "doubtful"
                conf_to_show = best_malign_conf
                box_to_draw = best_malign_box or best_benign_box

            # 2) Above 0.60 malign → Malign
            elif best_malign_conf > 0.60:
                label = "malign"
                conf_to_show = best_malign_conf
                box_to_draw = best_malign_box

            # 3) 0.30–0.60 malign → Doubtful
            elif 0.30 <= best_malign_conf <= 0.60:
                label = "doubtful"
                conf_to_show = best_malign_conf
                box_to_draw = best_malign_box

            # 4) Above 0.80 benign → Benign
            elif best_benign_conf > 0.80:
                label = "benign"
                conf_to_show = best_benign_conf
                box_to_draw = best_benign_box

            # 5) Fallback: something detected but not confident → Doubtful
            elif best_malign_conf > 0 or best_benign_conf > 0:
                label = "doubtful"
                # use whichever has higher confidence
                if best_malign_conf >= best_benign_conf:
                    conf_to_show = best_malign_conf
                    box_to_draw = best_malign_box
                else:
                    conf_to_show = best_benign_conf
                    box_to_draw = best_benign_box

            # Draw annotation according to final label
            if box_to_draw is not None:
                x1, y1, x2, y2 = map(int, box_to_draw)

                if label == "malign":
                    color = (0, 0, 255)  # Red
                    text = f"Malignant: {int(conf_to_show * 100)}%"
                elif label == "benign":
                    color = (0, 255, 0)  # Green
                    text = f"Benign: {int(conf_to_show * 100)}%"
                elif label == "doubtful":
                    color = (0, 255, 255)  # Yellow
                    pct = int((conf_to_show or 0.5) * 100)
                    text = f"Doubtful (borderline): ~{pct}%"
                else:
                    color = (255, 255, 255)
                    text = "No clear lesion"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    annotated,
                    "No lesion detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )

            return annotated, label, conf_to_show

        except Exception as e:
            print(f"Inference error: {e}")
            return frame_bgr, "error", None

    # ---------- IMAGE MODE (LOAD IMAGE) ----------

    def load_image(self):
        file_path = filedialog.askopenfilename(
            parent=self.window,
            title="Select Clinical Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        bgr = cv2.imread(file_path)
        if bgr is None:
            messagebox.showerror("Image Error", "Could not load the selected image.")
            return

        annotated_bgr, label, conf = self.run_inference_on_frame(bgr)

        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Store for PDF (uploaded image mode)
        self.last_original_image = Image.fromarray(original_rgb)
        self.last_annotated_image = Image.fromarray(annotated_rgb)
        self.last_prediction_label = label
        self.last_prediction_conf = conf

        self.show_image(self.last_annotated_image)

    def show_image(self, pil_image):
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        imgtk = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
        self.canvas.imgtk = imgtk

    # ---------- VIDEO MODE (NOW ALSO UPDATES last_* FOR PDF) ----------

    def update_video(self):
        if not self.running or self.vid is None:
            return

        ret, frame = self.vid.read()
        if not ret:
            self.window.after(30, self.update_video)
            return

        # Keep a copy of original frame for the report
        orig_bgr = frame.copy()

        annotated_bgr, label, conf = self.run_inference_on_frame(frame)

        # Convert to RGB for Tkinter
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        imgtk = ImageTk.PhotoImage(image=image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
        self.canvas.imgtk = imgtk

        # ---- KEY PART: store the current live-cam frame & prediction for PDF ----
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        self.last_original_image = Image.fromarray(orig_rgb)
        self.last_annotated_image = Image.fromarray(rgb)
        self.last_prediction_label = label
        self.last_prediction_conf = conf
        # ----------------------------------------------------------------------- #

        self.window.after(30, self.update_video)

    # ---------- PDF REPORT ----------

    def generate_pdf_report(self):
        # Works for both live-cam and uploaded-image modes,
        # because update_video() and load_image() both set last_*.
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader

        if self.last_original_image is None or self.last_annotated_image is None:
            messagebox.showinfo(
                "No Analysis",
                "No frame analyzed yet. Wait for live cam to show a frame or load an image first.",
            )
            return

        file_path = filedialog.asksaveasfilename(
            parent=self.window,
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save PDF Report",
            initialfile="Melanoma_Report.pdf",
        )
        if not file_path:
            return

        try:
            tmp_dir = tempfile.mkdtemp()
            orig_path = os.path.join(tmp_dir, "original.png")
            proc_path = os.path.join(tmp_dir, "processed.png")
            self.last_original_image.save(orig_path, format="PNG")
            self.last_annotated_image.save(proc_path, format="PNG")

            c = canvas.Canvas(file_path, pagesize=A4)
            page_width, page_height = A4

            margin_x = 2 * cm
            margin_y = 2 * cm

            # Title
            c.setFont("Helvetica-Bold", 18)
            c.drawString(margin_x, page_height - margin_y, "Melanoma Risk Analysis Report")

            c.setFont("Helvetica", 10)
            c.drawString(
                margin_x,
                page_height - margin_y - 15,
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            )
            c.drawString(
                margin_x,
                page_height - margin_y - 30,
                "Tool: Experimental YOLO-based lesion analysis (not for clinical use).",
            )

            current_y = page_height - margin_y - 60

            max_img_width = page_width - 2 * margin_x
            max_img_height = (page_height / 2) - margin_y - 40

            # Original image
            orig_reader = ImageReader(orig_path)
            orig_w, orig_h = orig_reader.getSize()
            scale_orig = min(max_img_width / orig_w, max_img_height / orig_h)
            draw_w_orig = orig_w * scale_orig
            draw_h_orig = orig_h * scale_orig

            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, current_y, "Original Frame / Clinical Image:")
            current_y -= 10 + draw_h_orig

            c.drawImage(
                orig_reader,
                margin_x,
                current_y,
                width=draw_w_orig,
                height=draw_h_orig,
                preserveAspectRatio=True,
                mask="auto",
            )

            # Processed image
            current_y -= 30
            proc_reader = ImageReader(proc_path)
            proc_w, proc_h = proc_reader.getSize()
            scale_proc = min(max_img_width / proc_w, max_img_height / proc_h)
            draw_w_proc = proc_w * scale_proc
            draw_h_proc = proc_h * scale_proc

            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, current_y, "Processed Image (Model Annotation):")
            current_y -= 10 + draw_h_proc

            c.drawImage(
                proc_reader,
                margin_x,
                current_y,
                width=draw_w_proc,
                height=draw_h_proc,
                preserveAspectRatio=True,
                mask="auto",
            )

            # New page
            c.showPage()
            page_width, page_height = A4
            margin_x = 2 * cm
            margin_y = 2 * cm
            current_y = page_height - margin_y

            # Prediction summary
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin_x, current_y, "Model Prediction Summary")
            current_y -= 25

            c.setFont("Helvetica", 11)
            if self.last_prediction_label == "malign":
                summary = f"AI interpretation: Malignant-appearing lesion (confidence ~ {int(self.last_prediction_conf * 100)}%)."
            elif self.last_prediction_label == "benign":
                if self.last_prediction_conf is not None:
                    summary = f"AI interpretation: Benign-appearing lesion (confidence ~ {int(self.last_prediction_conf * 100)}%)."
                else:
                    summary = "AI interpretation: Benign-appearing lesion."
            elif self.last_prediction_label == "doubtful":
                pct = int((self.last_prediction_conf or 0.5) * 100)
                summary = f"AI interpretation: Doubtful / borderline risk (malignant output around {pct}%)."
            elif self.last_prediction_label == "none":
                summary = "AI interpretation: No clear lesion detected in this frame/image."
            else:
                summary = "AI prediction could not be interpreted (error)."

            c.drawString(margin_x, current_y, summary)
            current_y -= 30

            # Strong disclaimer
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, current_y, "Important Disclaimer")
            current_y -= 20
            c.setFont("Helvetica", 10)

            disclaimer_lines = [
                "This PDF is generated by an experimental artificial intelligence tool.",
                "It is NOT a medical diagnosis and must NOT be used as a substitute for clinical evaluation.",
                "Regardless of the AI result (benign, malignant or doubtful), any suspicious lesion",
                "must be evaluated and followed by a dermatologist or qualified physician.",
            ]
            for line in disclaimer_lines:
                c.drawString(margin_x, current_y, line)
                current_y -= 14

            current_y -= 10

            # Melanoma info
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, current_y, "What is Melanoma?")
            current_y -= 18
            c.setFont("Helvetica", 10)

            melanoma_text = [
                "Melanoma is a type of skin cancer that develops from pigment-producing cells (melanocytes).",
                "It can spread early and may be life-threatening if not detected and treated in time.",
            ]
            for line in melanoma_text:
                c.drawString(margin_x, current_y, line)
                current_y -= 14

            current_y -= 10

            # ABCDE warning signs
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, current_y, "Warning Signs (ABCDE Checklist)")
            current_y -= 18
            c.setFont("Helvetica", 10)

            abcde_lines = [
                "A – Asymmetry: one half of the lesion looks different from the other.",
                "B – Border: irregular, blurred, or notched edges.",
                "C – Color: multiple colors (brown, black, red, white, blue) in one lesion.",
                "D – Diameter: larger than 6 mm (about a pencil eraser), though smaller melanomas exist.",
                "E – Evolution: any change in size, shape, color, or new symptoms (itching, bleeding).",
            ]
            for line in abcde_lines:
                c.drawString(margin_x, current_y, line)
                current_y -= 14

            current_y -= 10

            # Suggestions
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, current_y, "Suggested Next Steps")
            current_y -= 18
            c.setFont("Helvetica", 10)

            suggestion_lines = [
                "- If this lesion is new, growing, changing, or looks different from your other moles,",
                "  you should promptly consult a dermatologist.",
                "- Take clear, well-lit photos over time if advised by your physician.",
                "- Never delay medical evaluation based on AI or online tools.",
            ]
            for line in suggestion_lines:
                c.drawString(margin_x, current_y, line)
                current_y -= 14

            current_y -= 10

            followup = (
                "If your doctor suspects melanoma, they may recommend dermoscopy, digital monitoring, ",
                "or a biopsy to confirm the diagnosis."
            )
            c.drawString(margin_x, current_y, followup)

            c.showPage()
            c.save()

            messagebox.showinfo("Report Saved", f"PDF report saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Report Error", f"Could not generate report:\n{e}")

    # ---------- CLEANUP ----------

    def on_closing(self):
        self.running = False
        if hasattr(self, "vid") and self.vid is not None and self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    # Live cam starts automatically with video_source=0
    app = YOLOMelanomaApp(root, video_source=0)
    root.mainloop()
