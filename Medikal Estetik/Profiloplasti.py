import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ab = a - b
    cb = c - b
    dot_product = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return None
    angle_rad = np.arccos(dot_product / (norm_ab * norm_cb))
    return np.degrees(angle_rad)

def analyze_lateral_angles(image_path):
    """
    Detect landmarks, compute angles, and annotate the image with all 468 dots visible.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise ValueError("No face landmarks detected.")

        landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
        h, w, _ = image.shape
        points = lambda idx: (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))

        # Draw all 468 landmarks as dots on the image
        for idx, landmark in enumerate(landmarks):
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Draw dot in green
            cv2.putText(image, str(idx), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Define relevant points for angles
        forehead = points(10)
        nose_bridge = points(168)
        nose_tip = points(1)
        nasal_root = points(376)  # Nasal root for Nasolabial Angle
        chin_tip = points(152)

        # Calculate angles
        nasofrontal_angle = calculate_angle(forehead, nose_bridge, nose_tip)
        nasolabial_angle = calculate_angle(nose_tip, nasal_root, chin_tip)
        nasomental_angle = calculate_angle(nose_bridge, nose_tip, chin_tip)
        nasofacial_angle = 180 - nasofrontal_angle  # Derived from Nasofrontal Angle

        angles = {
            "Nasofrontal Angle": nasofrontal_angle,
            "Nasolabial Angle": nasolabial_angle,
            "Nasomental Angle": nasomental_angle,
            "Nasofacial Angle": nasofacial_angle,
        }

        # Draw lines between relevant points
        def draw_line_with_text(img, pt1, pt2, text=None, text_pos=None):
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            if text and text_pos:
                cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw Nasofrontal, Nasolabial, and Nasomental lines
        draw_line_with_text(image, forehead, nose_bridge, "NF", (forehead[0] + 10, forehead[1] - 20))
        draw_line_with_text(image, nose_bridge, nose_tip, "NF", (nose_bridge[0] + 10, nose_bridge[1] - 20))
        draw_line_with_text(image, nose_tip, nasal_root, "NL", (nose_tip[0] + 10, nose_tip[1] - 20))
        draw_line_with_text(image, nasal_root, chin_tip, None)  # No label for Nasolabial extension
        draw_line_with_text(image, nose_tip, chin_tip, "NM", (nose_tip[0] + 10, nose_tip[1] + 20))  # Nasomental line

        # Annotate angles in the top-left corner
        y_offset = 20  # Start at 20 pixels from the top
        for angle_name, angle_value in angles.items():
            text = f"{angle_name}: {angle_value:.2f}°" if angle_value else f"{angle_name}: N/A"
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30  # Move down for the next angle

        # Save processed image
        output_path = image_path.replace('.jpg', '_processed.jpg').replace('.png', '_processed.png')
        cv2.imwrite(output_path, image)

        return angles, output_path


class FaceAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lateral Face Profiloplasty Analysis")
        self.root.geometry("800x600")

        self.uploaded_image_path = None
        self.processed_image_path = None

        # Widgets
        self.label = tk.Label(root, text="Upload an Image for Analysis", font=("Arial", 16))
        self.label.pack(pady=20)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image, font=("Arial", 14))
        self.upload_button.pack(pady=10)

        self.result_frame = tk.Frame(root)
        self.result_frame.pack(pady=20)

        self.image_label = tk.Label(self.result_frame)
        self.image_label.pack()

        self.results_text = tk.Text(self.result_frame, height=10, width=50, state=tk.DISABLED)
        self.results_text.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Processed Image", command=self.save_image, font=("Arial", 14), state=tk.DISABLED)
        self.save_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        try:
            self.uploaded_image_path = file_path
            angles, self.processed_image_path = analyze_lateral_angles(file_path)
            self.display_results(angles, self.processed_image_path)
            self.save_button.config(state=tk.NORMAL)
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, angles, processed_image_path):
        # Display processed image
        img = Image.open(processed_image_path)
        img = img.resize((400, 400))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        # Display angles
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        for angle_name, angle_value in angles.items():
            self.results_text.insert(tk.END, f"{angle_name}: {angle_value:.2f}°\n")
        self.results_text.config(state=tk.DISABLED)

    def save_image(self):
        if not self.processed_image_path:
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if save_path:
            original_image = cv2.imread(self.processed_image_path)
            cv2.imwrite(save_path, original_image)
            messagebox.showinfo("Saved", f"Image saved to {save_path}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalysisApp(root)
    root.mainloop()
