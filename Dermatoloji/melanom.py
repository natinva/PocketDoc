import tkinter as tk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
from pathlib import Path

class YOLOSegmentationApp:
    def __init__(self, window, video_source=0, model_path=None):
        self.window = window
        self.window.title("YOLOv12 Live Inference")
        self.video_source = video_source

        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "Modeller" / "melanom.pt"

        # Load the YOLO model
        try:
            self.model = YOLO(str(model_path))
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model: {e}")
            self.window.destroy()
            return

        # Open video source
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            messagebox.showerror("Video Error", "Unable to open video source")
            self.window.destroy()
            return

        # Retrieve video dimensions
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup a canvas to display video frames
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Optional: add a quit button for exiting the application
        self.btn_quit = tk.Button(window, text="Quit", width=50, command=self.on_closing)
        self.btn_quit.pack(anchor=tk.CENTER, expand=True)

        # Create a dedicated label at the bottom of the window for the caution text.
        self.caution_label = tk.Label(window,
                                      text="Always see an expert dermatologist for those lesions.",
                                      font=("Helvetica", 12))
        self.caution_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Set frame update delay (in milliseconds) and start the update loop.
        self.delay = 15
        self.update()

        # Set window closing protocol.
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            try:
                # Run inference on the frame using the YOLO model.
                results = self.model(frame)
                boxes = results[0].boxes  # Detected objects

                selected_box = None
                selected_conf = 0  # Highest confidence found
                selected_label = None

                # Look for "malign" detections (assumed class 1) with confidence >= 80% (0.8)
                for box in boxes:
                    conf = box.conf.item()  # confidence as float
                    cls = int(box.cls.item())  # predicted class (1 for malign, 0 for benign)
                    if cls == 1 and conf >= 0.7:
                        if selected_box is None or conf > selected_conf:
                            selected_box = box.xyxy.cpu().numpy()[0]
                            selected_conf = conf
                            selected_label = "malign"

                # If no high-confidence malignant box is found, pick a candidate for benign display
                if selected_box is None and len(boxes) > 0:
                    for box in boxes:
                        conf = box.conf.item()
                        if selected_box is None or conf > selected_conf:
                            selected_box = box.xyxy.cpu().numpy()[0]
                            selected_conf = conf
                            selected_label = "benign"

                # If a box is selected, draw its bounding box and the corresponding label
                if selected_box is not None:
                    x1, y1, x2, y2 = map(int, selected_box)
                    if selected_label == "malign":
                        color = (0, 0, 255)  # Red for malignant
                        text = f"Malign: {int(selected_conf * 100)}%"
                    else:
                        color = (0, 255, 0)  # Green for benign
                        text = "Benign"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except Exception as e:
                print(f"Inference error: {e}")

            # Convert the frame from BGR to RGB for Tkinter display
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(annotated_frame)
            imgtk = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
            self.canvas.imgtk = imgtk  # Prevent garbage collection

        # Schedule the next frame update
        self.window.after(self.delay, self.update)

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    # The model path defaults to Modeller/melanom.pt relative to this file.
    app = YOLOSegmentationApp(root, video_source=0)
    root.mainloop()
