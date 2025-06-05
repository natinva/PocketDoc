import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------
# Load your YOLOv8 segmentation model
# ---------------------------
MODEL_PATH = "/Users/avnitan/Downloads/Fossa/best (3).pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")


def rotate_points(points, M):
    """
    Applies an affine transform M (2x3) to an Nx2 array of (x,y) points.
    Returns the transformed Nx2 array.
    """
    pts = np.array([points], dtype=np.float32)  # shape (1, N, 2)
    pts_rot = cv2.transform(pts, M)            # shape (1, N, 2)
    return pts_rot[0]


def process_image(img_pil):
    """
    1. Convert PIL->OpenCV BGR.
    2. Run YOLO segmentation for "humerus" and "fossa".
    3. Overlay masks for visualization (optional).
    4. Extract & morphologically clean the humerus mask.
    5. Compute PCA => principal axis => rotation matrix => rotate humerus (and fossa if present).
    6. In rotated space:
       - Find bottom corners of humerus
       - If fossa is present, find top of fossa - 5 px
         else do 90° cross in humerus-axis space.
       - Generate line endpoints for the "X".
    7. Transform line endpoints back to original coordinates & draw on the original image.
    """
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1) Run inference
    results = model(img_np)
    result = results[0]
    if result.masks is None:
        return img_cv, "No segmentation masks found."

    masks = result.masks

    # Track best humerus & fossa polygons
    humerus_polygon = None
    humerus_conf = 0.0
    fossa_polygon = None
    fossa_conf = 0.0

    # 2) Overlay segmentation masks (optional) & store polygons
    for i, seg_xy in enumerate(masks.xy):
        cls_idx = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        label = model.names[cls_idx].lower()  # "humerus" or "fossa"
        polygon = seg_xy.astype(np.int32)

        # Visualization overlay
        overlay = img_cv.copy()
        if label == "humerus":
            color = (0, 255, 255)  # cyan
        elif label == "fossa":
            color = (0, 0, 255)    # red
        else:
            color = (128, 128, 128)
        cv2.fillPoly(overlay, [polygon], color)
        alpha = 0.5
        img_cv = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)

        # Keep best polygons by confidence
        if label == "humerus" and conf > humerus_conf:
            humerus_conf = conf
            humerus_polygon = polygon
        elif label == "fossa" and conf > fossa_conf:
            fossa_conf = conf
            fossa_polygon = polygon

    if humerus_polygon is None:
        return img_cv, "Humerus segmentation not found."

    # 3) Morphological cleanup on humerus
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [humerus_polygon], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Extract humerus points
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img_cv, "No humerus pixels after cleanup."
    humerus_points = np.vstack([xs, ys]).T  # shape (N,2)

    # 4) PCA to find humerus principal axis
    mean, eigenvectors = cv2.PCACompute(humerus_points.astype(np.float32), mean=np.array([]))
    # principal axis is eigenvectors[0]
    principal_axis = eigenvectors[0]
    # angle between principal axis and horizontal axis
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    # we want the humerus axis to be vertical => rotate by (90 - angle)
    rotation_angle_degs = np.degrees(np.pi/2 - angle)

    # Center of rotation is the humerus centroid
    center = (mean[0][0], mean[0][1])  # shape (1,2) => (x,y)
    M = cv2.getRotationMatrix2D(center, rotation_angle_degs, 1.0)
    # Invert for going back to original coords
    M_inv = cv2.invertAffineTransform(M)

    # Rotate humerus points
    humerus_points_rot = rotate_points(humerus_points, M)
    # (Optional) rotate fossa polygon if present
    fossa_points_rot = None
    if fossa_polygon is not None:
        fossa_points_rot = rotate_points(fossa_polygon, M)

    # 5) In rotated space, find the "bottom corners" of the humerus
    # We'll define "bottom" as the lower 10% in y
    y_max = np.max(humerus_points_rot[:, 1])
    y_min = np.min(humerus_points_rot[:, 1])
    height = y_max - y_min
    threshold = y_max - 0.1 * height  # bottom 10%
    bottom_region = humerus_points_rot[humerus_points_rot[:, 1] >= threshold]
    if len(bottom_region) == 0:
        bottom_region = humerus_points_rot  # fallback if humerus is small

    # left corner => min x in bottom region
    # right corner => max x in bottom region
    left_corner_rot = bottom_region[np.argmin(bottom_region[:, 0])]
    right_corner_rot = bottom_region[np.argmax(bottom_region[:, 0])]

    # 6) Intersection point in rotated space
    if fossa_points_rot is not None and len(fossa_points_rot) > 0:
        # top of fossa => min y
        fossa_top_rot = np.min(fossa_points_rot[:, 1])
        intersection_y = fossa_top_rot - 5
        intersection_x = (left_corner_rot[0] + right_corner_rot[0]) / 2
        intersection_rot = np.array([intersection_x, intersection_y], dtype=np.float32)
    else:
        # no fossa => lines at ±45° => 90° cross in humerus-axis space
        # We'll define lines from the corners up to an intersection
        # that is "distance/2" above the corners in y
        # (like the old bounding box approach, but in rotated coords)
        width = right_corner_rot[0] - left_corner_rot[0]
        intersection_x = (left_corner_rot[0] + right_corner_rot[0]) / 2
        intersection_y = left_corner_rot[1] - (width / 2)
        intersection_rot = np.array([intersection_x, intersection_y], dtype=np.float32)

    # 7) Transform these points back to original coordinates
    left_anchor = rotate_points([left_corner_rot], M_inv)[0]
    right_anchor = rotate_points([right_corner_rot], M_inv)[0]
    intersection_pt = rotate_points([intersection_rot], M_inv)[0]

    left_anchor = tuple(int(v) for v in left_anchor)
    right_anchor = tuple(int(v) for v in right_anchor)
    intersection_pt = tuple(int(v) for v in intersection_pt)

    # 8) Draw lines in the original image
    overlay = img_cv.copy()
    line_color = (255, 0, 0)  # blue
    thickness = 5
    cv2.line(overlay, left_anchor, intersection_pt, line_color, thickness)
    cv2.line(overlay, right_anchor, intersection_pt, line_color, thickness)
    alpha_line = 0.8
    img_cv = cv2.addWeighted(overlay, alpha_line, img_cv, 1 - alpha_line, 0)

    # Info message
    msg = f"Humerus (conf={humerus_conf:.2f}). "
    if fossa_polygon is not None:
        msg += f"Fossa (conf={fossa_conf:.2f}). "
        msg += "Lines aligned with humerus axis (intersection above fossa)."
    else:
        msg += "No fossa. Lines at 90° in humerus-axis space."

    return img_cv, msg


def load_and_process_image():
    file_path = filedialog.askopenfilename(filetypes=[
        ("JPEG files", "*.jpg *.jpeg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ])
    if not file_path:
        return
    try:
        img_pil = Image.open(file_path).convert("RGB")
    except Exception as e:
        messagebox.showerror("Error", f"Could not open image:\n{e}")
        return

    annotated_cv_img, info_msg = process_image(img_pil)
    # Convert BGR -> RGB -> PIL
    annotated_rgb = cv2.cvtColor(annotated_cv_img, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)
    annotated_pil.thumbnail((800, 600))
    annotated_tk = ImageTk.PhotoImage(annotated_pil)

    image_label.config(image=annotated_tk)
    image_label.image = annotated_tk  # keep a reference
    text_details.delete("1.0", tk.END)
    text_details.insert(tk.END, info_msg)


# ---------------------------
# Tkinter GUI
# ---------------------------
root = tk.Tk()
root.title("Aligned X-wire with Humerus Axis")
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

btn_load = tk.Button(frame, text="Load Image", command=load_and_process_image, font=("Arial", 12))
btn_load.pack(pady=5)

image_label = tk.Label(frame)
image_label.pack(pady=5)

text_details = tk.Text(frame, height=6, width=80, font=("Arial", 10))
text_details.pack(pady=5)

root.mainloop()
