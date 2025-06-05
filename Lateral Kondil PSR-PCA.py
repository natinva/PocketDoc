import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import math

# Global variables
fracture_points = []
image = None
image_copy = None

# Function to open file dialog and load the image
def load_image():
    global image, image_copy
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Error", "Image could not be loaded!")
        else:
            image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            messagebox.showinfo("Success", "Image loaded successfully!")
    else:
        messagebox.showwarning("Warning", "No image selected!")

# Mouse callback function to capture two points for the fracture site
def select_fracture_site(event, x, y, flags, param):
    global fracture_points
    if event == cv2.EVENT_LBUTTONDOWN and len(fracture_points) < 2:
        fracture_points.append((x, y))
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)  # Mark the point
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
    return angle_degrees if angle_degrees <= 180 else 360 - angle_degrees

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

# Function to preprocess the image
def preprocess_image(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)
    return cv2.Canny(thresholded_image, 50, 150, apertureSize=3)

# Function to extend lines to the image boundary
def extend_line_to_boundary(line, width, height):
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1

    if dx == 0:  # Vertical line
        return [(x1, 0), (x1, height - 1)]
    if dy == 0:  # Horizontal line
        return [(0, y1), (width - 1, y1)]

    # Calculate intersection with image boundaries
    points = []
    for x, y in [(0, y1 - x1 * dy // dx), (width - 1, y1 + (width - 1 - x1) * dy // dx),
                 (x1 - y1 * dx // dy, 0), (x1 + (height - 1 - y1) * dx // dy, height - 1)]:
        if 0 <= x < width and 0 <= y < height:
            points.append((int(x), int(y)))
    return points[:2]

# Function to add results text below the image
def display_results(image, psr, pca):
    height, width, _ = image.shape
    padding = 100
    result_image = np.zeros((height + padding, width, 3), dtype=np.uint8)
    result_image[:height, :, :] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_image, f"PSR: {psr:.2f}", (10, height + 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result_image, f"PCA: {pca:.2f} deg", (10, height + 70), font, 0.7, (255, 255, 255), 2)
    return result_image

# Function to detect K-wires and calculate the required measurements
def detect_kwires_and_calculate():
    global fracture_points, image_copy
    if image is None:
        messagebox.showwarning("Warning", "You must load an image first!")
        return

    edges = preprocess_image(image)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) < 2:
        messagebox.showerror("Error", "Could not detect two K-wires!")
        return

    lines = [line[0] for line in lines]
    valid_pair = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            angle = calculate_angle(lines[i], lines[j])
            if angle >= 20:  # Divergent K-wires angle threshold
                valid_pair = (lines[i], lines[j])
                break
        if valid_pair:
            break

    if valid_pair is None:
        messagebox.showerror("Error", "No valid pair of K-wires found!")
        return

    line1, line2 = valid_pair
    height, width = image_copy.shape[:2]

    # Extend K-wires to the image boundaries
    extended_line1 = extend_line_to_boundary(line1, width, height)
    extended_line2 = extend_line_to_boundary(line2, width, height)

    # Draw extended K-wires in yellow
    cv2.line(image_copy, extended_line1[0], extended_line1[1], (0, 255, 255), 2)
    cv2.line(image_copy, extended_line2[0], extended_line2[1], (0, 255, 255), 2)

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

            # Create and annotate fracture image
            fracture_image = image_copy.copy()
            cv2.line(fracture_image, a, b, (255, 0, 0), 2)  # Blue fracture line
            fracture_image = display_results(fracture_image, psr, pca)

            # Show the results
            cv2.imshow("Fracture Line with Results", fracture_image)
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

