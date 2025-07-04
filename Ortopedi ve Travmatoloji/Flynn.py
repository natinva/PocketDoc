import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops

class AutoCarryingAngleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Flynn Carrying-Angle")
        self.orig = None
        self.display = None
        self.photo = None

        # Canvas
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Controls
        frm = tk.Frame(root); frm.pack(fill=tk.X)
        tk.Button(frm, text="Load Image",    command=self.load_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(frm, text="Compute Angle", command=self.compute_angle).pack(side=tk.LEFT, padx=5)
        self.lbl = tk.Label(frm, text="Angle: N/A"); self.lbl.pack(side=tk.LEFT, padx=20)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.orig    = gray
        self.display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self._show(self.display)
        self.lbl.config(text="Angle: N/A")

    def _show(self, bgr):
        self.canvas.delete("all")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im  = Image.fromarray(rgb)
        self.photo   = ImageTk.PhotoImage(im)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def compute_angle(self):
        if self.orig is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return

        # 1) Threshold + cleanup
        blur = cv2.GaussianBlur(self.orig, (5,5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = th > 0
        mask = remove_small_objects(mask, min_size=500)  # drop speckles

        # 2) Skeletonize
        skel = skeletonize(mask)
        # Label connected skeleton segments
        lbl = label(skel)
        props = sorted(regionprops(lbl), key=lambda r: r.area, reverse=True)
        if len(props) < 2:
            messagebox.showerror("Error", "Could not isolate two shaft segments.")
            return

        # Take the two largest components
        coords_h = props[0].coords  # humerus
        coords_u = props[1].coords  # ulna

        # 3) PCA on each to get axis
        def pca_axis(pts):
            # pts are [[row, col],...]
            mean = pts.mean(axis=0)
            cov  = np.cov((pts-mean).T)
            vals, vecs = np.linalg.eigh(cov)
            principal = vecs[:, np.argmax(vals)]  # [dy,dx]
            # convert to image [dx,dy] and normalize
            vec = np.array([principal[1], principal[0]])
            vec /= np.linalg.norm(vec)
            origin = np.array([mean[1], mean[0]])  # (x=col, y=row)
            return origin, vec

        o_h, v_h = pca_axis(coords_h)
        o_u, v_u = pca_axis(coords_u)

        # 4) Compute signed angle
        dot = np.clip(np.dot(v_h, v_u), -1, 1)
        ang = np.degrees(np.arccos(dot))
        if (v_h[0]*v_u[1] - v_h[1]*v_u[0]) < 0:
            ang = -ang

        # 5) Draw axes & annotate
        disp = self.display.copy()
        L    = max(disp.shape[:2])
        for (origin, vec, col) in ((o_h, v_h, (0,255,0)), (o_u, v_u, (255,0,0))):
            x0,y0 = origin.astype(int)
            x1,y1 = int(x0+vec[0]*L), int(y0+vec[1]*L)
            x2,y2 = int(x0-vec[0]*L), int(y0-vec[1]*L)
            cv2.line(disp, (x1,y1), (x2,y2), col, 2)

        cv2.putText(
            disp, f"{ang:+.1f}°",
            tuple(o_h.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA
        )

        self._show(disp)
        self.lbl.config(text=f"Angle: {ang:.1f}°")


if __name__ == "__main__":
    root = tk.Tk()
    AutoCarryingAngleApp(root)
    root.mainloop()
