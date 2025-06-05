import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import math

# Global variables to store points for humerus and forearm axes
humerus_points = []
forearm_points = []
image = None
image_copy = None

# Function to calculate angle between two lines
def calculate_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    v1 = (x2 - x1, y2 - y1)
    v2 = (x4 - x3, y4 - y3)
    angle_radians = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    angle_degrees = np.degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees += 360
    if angle_degrees > 180:
        angle_degrees = 360 - angle_degrees
    return angle_degrees

# Flynn score calculation based on carrying angle loss
def calculate_flynn_score(carrying_angle_loss):
    if carrying_angle_loss < 5:
        return "Excellent"
    elif carrying_angle_loss < 10:
        return "Good"
    elif carrying_angle_loss < 15:
        return "Fair"
    else:
        return "Poor"

# Mouse callback function to select points
def select_points(event, x, y, flags, param):
    global humerus_points, forearm_points, image_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(humerus_points) < 2:
            humerus_points.append((x, y))
            cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Humerus and Forearm Axes", image_copy)
            if len(humerus_points) == 2:  # After the second click for humerus
                instructions_var.set("Now click two points for the forearm axis.")
        elif len(forearm_points) < 2:
            forearm_points.append((x, y))
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Humerus and Forearm Axes", image_copy)

        # When both axes are selected, draw lines and calculate the angle
        if len(humerus_points) == 2 and len(forearm_points) == 2:
            cv2.line(image_copy, humerus_points[0], humerus_points[1], (255, 0, 0), 2)
            cv2.line(image_copy, forearm_points[0], forearm_points[1], (0, 255, 0), 2)
            cv2.imshow("Select Humerus and Forearm Axes", image_copy)

            # Calculate carrying angle
            carrying_angle = calculate_angle(humerus_points[0] + humerus_points[1], forearm_points[0] + forearm_points[1])
            angle_var.set(f"Carrying Angle: {carrying_angle:.2f} degrees")

            # Flynn score calculation (assume normal carrying angle is 15 degrees)
            normal_carrying_angle = 15
            carrying_angle_loss = abs(normal_carrying_angle - carrying_angle)
            flynn_score = calculate_flynn_score(carrying_angle_loss)

            # Display results and Flynn score classification
            report = f"""
            Carrying Angle Measured: {carrying_angle:.2f} degrees
            Ideal Carrying Angle: {normal_carrying_angle:.2f} degrees
            Loss in Carrying Angle: {carrying_angle_loss:.2f} degrees
            Flynn Score Classification: {flynn_score}
            """
            report_var.set(report)

# Load image function
def load_image():
    global image, image_copy, humerus_points, forearm_points
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
    if file_path:
        image = cv2.imread(file_path)
        image_copy = image.copy()
        humerus_points = []
        forearm_points = []
        instructions_var.set("Click two points to mark the humerus axis.")
        cv2.imshow("Select Humerus and Forearm Axes", image_copy)
        cv2.setMouseCallback("Select Humerus and Forearm Axes", select_points)
    else:
        messagebox.showwarning("Warning", "No image selected.")

# Tkinter GUI setup
root = tk.Tk()
root.title("Elbow Carrying Angle and Flynn Score Calculator")

# Instruction label
instructions_var = tk.StringVar()
instructions_var.set("Load an image to start.")
instructions_label = tk.Label(root, textvariable=instructions_var)
instructions_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Result display for carrying angle and Flynn score
angle_var = tk.StringVar()
report_var = tk.StringVar()

angle_label = tk.Label(root, textvariable=angle_var)
angle_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

report_label = tk.Label(root, textvariable=report_var, justify="left")
report_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Load Image button
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()
