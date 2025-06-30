import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import math

# Global variable to store fracture points
fracture_points = []

# Function to open file dialog and load the image
def load_image():
    global image, image_copy
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image_copy = image.copy()
        if image is None:
            messagebox.showerror("Error", "Image could not be loaded!")
        else:
            messagebox.showinfo("Success", "Image loaded successfully!")
    else:
        messagebox.showwarning("Warning", "No image selected!")

# Mouse callback function to capture two points for fracture site
def select_fracture_site(event, x, y, flags, param):
    global fracture_points
    if event == cv2.EVENT_LBUTTONDOWN and len(fracture_points) < 2:
        fracture_points.append((x, y))
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)  # Visual feedback on selected points
        cv2.imshow("Select Fracture Site", image_copy)

# Function to calculate the angle between two lines
def calculate_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    v1 = (x2 - x1, y2 - y1)
    v2 = (x4 - x3, y4 - y3)
    angle_radians = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    angle_degrees = np.degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees += 360

    if angle_degrees <= 180:
        angle_degrees = 180 - angle_degrees
    else:
        angle_degrees = 360 - angle_degrees
    return angle_degrees

# Function to calculate the intersection between a fracture line and the K-wire lines
def calculate_intersection(line, a, b):
    x1, y1, x2, y2 = line
    denominator = (x1 - x2) * (a[1] - b[1]) - (y1 - y2) * (a[0] - b[0])
    if denominator == 0:
        return None  # Parallel lines
    t = ((x1 - a[0]) * (a[1] - b[1]) - (y1 - a[1]) * (a[0] - b[0])) / denominator
    intersection_x = x1 + t * (x2 - x1)
    intersection_y = y1 + t * (y2 - y1)
    return int(intersection_x), int(intersection_y)
# Function to calculate the length between two points
def calculate_distance(point1, point2):
    return np.linalg.norm([point1[0] - point2[0], point1[1] - point2[1]])

# Function to extend a line
def extend_line(line, image_shape, extend_length=1000):
    x1, y1, x2, y2 = line
    vx, vy = x2 - x1, y2 - y1
    length = np.sqrt(vx ** 2 + vy ** 2)
    vx, vy = vx / length, vy / length
    x1_ext = int(x1 - vx * extend_length)
    y1_ext = int(y1 - vy * extend_length)
    x2_ext = int(x2 + vx * extend_length)
    y2_ext = int(y2 + vy * extend_length)
    return x1_ext, y1_ext, x2_ext, y2_ext

# Function to detect K-wires and calculate the required measurements
def detect_kwires_and_calculate():
    global fracture_points
    if image is None:
        messagebox.showwarning("Warning", "You must load an image first!")
        return

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresholded_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) < 2:
        messagebox.showerror("Error", "Could not detect two K-wires!")
    else:
        lines = [line[0] for line in lines]
        lines = sorted(lines, key=lambda line: np.linalg.norm((line[2] - line[0], line[3] - line[1])), reverse=True)
        valid_pair = None
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                angle = calculate_angle(lines[i], lines[j])
                if angle >= 20:
                    valid_pair = (lines[i], lines[j])
                    break
            if valid_pair:
                break

        if valid_pair is None:
            messagebox.showerror("Error", "No valid pair of K-wires found!")
        else:
            line1, line2 = valid_pair
            extended_line1 = extend_line(line1, image.shape)
            extended_line2 = extend_line(line2, image.shape)

            # Draw the K-wire lines in yellow
            cv2.line(image_copy, (extended_line1[0], extended_line1[1]), (extended_line1[2], extended_line1[3]), (0, 255, 255), 2)
            cv2.line(image_copy, (extended_line2[0], extended_line2[1]), (extended_line2[2], extended_line2[3]), (0, 255, 255), 2)

            # Show the image and allow user to select fracture points
            cv2.imshow("Select Fracture Site", image_copy)
            cv2.setMouseCallback("Select Fracture Site", select_fracture_site)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if len(fracture_points) == 2:
                a, b = fracture_points

                # Calculate intersections and PSR
                x1_intersection = calculate_intersection(line1, a, b)
                x2_intersection = calculate_intersection(line2, a, b)

                if x1_intersection and x2_intersection:
                    distance_ab = calculate_distance(a, b)
                    distance_xy = calculate_distance(x1_intersection, x2_intersection)
                    psr = distance_xy / distance_ab

                    # Calculate PCA
                    pca = calculate_angle(line1, line2)

                    # Display results on image
                    height, width = image.shape
                    cv2.putText(image, f"PSR: {psr:.2f}", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(image, f"PCA: {pca:.2f} deg", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Show image with results
                    cv2.imshow("K-Wire Measurement", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    messagebox.showerror("Error", "Could not calculate intersections!")
            else:
                messagebox.showerror("Error", "Fracture site not selected properly!")

# Main Window Setup
root = tk.Tk()
root.title("K-Wire Measurement and Calculation")

# Load Image Button
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

# Detect and Calculate Button
calculate_button = tk.Button(root, text="Detect K-Wires and Calculate", command=detect_kwires_and_calculate)
calculate_button.pack(pady=10)

# Start GUI
root.mainloop()
