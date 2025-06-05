import tkinter as tk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk


class YOLOSegmentationApp:
    def __init__(self, window, video_source=0, model_path="/Users/avnitan/Downloads/Modeller/Medical Aesthetic/Dark Circles.pt"):
        self.window = window
        self.window.title("YOLOv8 Segmentation Live Inference")
        self.video_source = video_source

        # Load the YOLOv8 segmentation model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model: {e}")
            self.window.destroy()
            return

        # Open video source (webcam by default)
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            messagebox.showerror("Video Error", "Unable to open video source")
            self.window.destroy()
            return

        # Retrieve video dimensions
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set up a Tkinter canvas to display the video frames
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Optional: add a quit button to safely exit the application
        self.btn_quit = tk.Button(window, text="Quit", width=50, command=self.on_closing)
        self.btn_quit.pack(anchor=tk.CENTER, expand=True)

        # Start the video loop
        self.delay = 15  # milliseconds between frame updates
        self.update()

        # Ensure proper cleanup on closing the window
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Run YOLOv8 inference on the current frame
            try:
                results = self.model(frame)  # model inference
                # Get the annotated image from the first result
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"Inference error: {e}")
                annotated_frame = frame

            # Convert BGR (OpenCV) to RGB (PIL)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(annotated_frame)
            imgtk = ImageTk.PhotoImage(image=image)

            # Display the image in the canvas
            self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
            self.canvas.imgtk = imgtk  # keep a reference to prevent garbage collection

        # Schedule the next frame update
        self.window.after(self.delay, self.update)

    def on_closing(self):
        # Release video capture and destroy the Tkinter window
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    # Update "your_model.pt" with the path to your trained YOLOv8 segmentation model file.
    app = YOLOSegmentationApp(root, video_source=0, model_path="/Users/avnitan/Downloads/Modeller/Medical Aesthetic/Dark Circles.pt")
    root.mainloop()
