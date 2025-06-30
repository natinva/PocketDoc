import cv2
import mediapipe as mp
import numpy as np
import math
from tkinter import *
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk

# Pose setup using Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to detect pose and landmarks
def detectPose(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return frame, results.pose_landmarks.landmark
    return frame, None


# Function to classify pose and calculate angles for both arms
def classifyPose(landmarks, frame):
    # Swapping the labels for left and right to correct the mirror effect
    # Corrected: Left becomes Right, and Right becomes Left
    # Get coordinates for right joints (actually the left side due to the mirror effect)
    right_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
    right_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
    right_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]

    # Get coordinates for left joints (actually the right side due to the mirror effect)
    left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
    left_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
    left_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]

    # Calculate right elbow angle (which is actually the left elbow in mirror view)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    cv2.putText(frame, f'Right Elbow Angle: {int(right_elbow_angle)}', tuple(np.multiply(right_elbow, [1, 1]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Calculate left elbow angle (which is actually the right elbow in mirror view)
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    cv2.putText(frame, f'Left Elbow Angle: {int(left_elbow_angle)}', tuple(np.multiply(left_elbow, [1, 1]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, left_elbow_angle, right_elbow_angle


# Tkinter window class to handle the webcam and image display
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0  # Default webcam

        # Set desired camera resolution (example 640x480)
        self.width = 640
        self.height = 480

        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.canvas = Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Buttons to start video, stop video, and load image
        self.btn_start = Button(window, text="Start Webcam", width=50, command=self.start_video)
        self.btn_start.pack(anchor=CENTER, expand=True)

        self.btn_stop = Button(window, text="Stop Webcam", width=50, command=self.stop_video)
        self.btn_stop.pack(anchor=CENTER, expand=True)

        self.btn_image = Button(window, text="Load Image", width=50, command=self.load_image)
        self.btn_image.pack(anchor=CENTER, expand=True)

        # Buttons to capture flexion and extension angles
        self.btn_extension = Button(window, text="Capture Extension", width=50, command=self.capture_extension)
        self.btn_extension.pack(anchor=CENTER, expand=True)

        self.btn_flexion = Button(window, text="Capture Flexion", width=50, command=self.capture_flexion)
        self.btn_flexion.pack(anchor=CENTER, expand=True)

        self.running = False
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, model_complexity=1)

        # Hold a reference to the loaded image to prevent garbage collection
        self.imgtk = None

        # Initialize variables to store extension and flexion data
        self.extension_image = None
        self.flexion_image = None
        self.extension_left_angle = 0
        self.flexion_left_angle = 0
        self.extension_right_angle = 0
        self.flexion_right_angle = 0

        self.window.mainloop()

    def start_video(self):
        self.running = True
        self.update()

    def stop_video(self):
        self.running = False

    def update(self):
        if self.running:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Flip for selfie view

                # Resize frame to fit display
                frame = cv2.resize(frame, (self.width, self.height))

                # Detect pose and calculate angles
                frame, landmarks = detectPose(frame, self.pose)
                if landmarks:
                    frame, _, _ = classifyPose(landmarks, frame)

                # Convert frame to Tkinter-compatible image format
                self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.photo))

                self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
                self.window.after(10, self.update)

    def load_image(self):
        # Stop webcam if running
        self.running = False

        # Open file dialog without file type filters (to avoid macOS crash)
        image_path = filedialog.askopenfilename()
        if image_path:
            # Read the image and process it
            frame = cv2.imread(image_path)
            frame = cv2.resize(frame, (self.width, self.height))

            # Detect pose in the image
            frame, landmarks = detectPose(frame, self.pose)
            if landmarks:
                frame, _, _ = classifyPose(landmarks, frame)

            # Convert the image to display in Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.imgtk = ImageTk.PhotoImage(
                image=Image.fromarray(frame_rgb))  # Store in self.imgtk to avoid garbage collection

            self.canvas.create_image(0, 0, image=self.imgtk, anchor=NW)
            self.window.update()

    def capture_extension(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame, landmarks = detectPose(frame, self.pose)
            if landmarks:
                frame, left_elbow_angle, right_elbow_angle = classifyPose(landmarks, frame)
                # Store the extension image and both elbow angles
                self.extension_image = frame
                self.extension_left_angle = left_elbow_angle
                self.extension_right_angle = right_elbow_angle
                # Save the image with extension angle
                cv2.imwrite(f'extension_angle_{int(right_elbow_angle)}.png', frame)

    def capture_flexion(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame, landmarks = detectPose(frame, self.pose)
            if landmarks:
                frame, left_elbow_angle, right_elbow_angle = classifyPose(landmarks, frame)
                # Store the flexion image and both elbow angles
                self.flexion_image = frame
                self.flexion_left_angle = left_elbow_angle
                self.flexion_right_angle = right_elbow_angle
                # Save the image with flexion angle
                cv2.imwrite(f'flexion_angle_{int(right_elbow_angle)}.png', frame)

            # After capturing both flexion and extension, show them side by side
            self.show_side_by_side()

    def show_side_by_side(self):
        if self.extension_image is not None and self.flexion_image is not None:
            # Create a new window to show the images side by side
            side_by_side_window = Toplevel(self.window)
            side_by_side_window.title("Extension vs Flexion")

            # Combine both images side by side and resize to fit screen
            combined_image = np.hstack((self.extension_image, self.flexion_image))
            screen_width = self.window.winfo_screenwidth()
            scale_ratio = screen_width / combined_image.shape[1]  # Scaling factor to fit width
            new_height = int(combined_image.shape[0] * scale_ratio)
            resized_image = cv2.resize(combined_image, (screen_width, new_height))

            # Convert the combined image to Tkinter-compatible format
            combined_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            imgtk_combined = ImageTk.PhotoImage(image=Image.fromarray(combined_rgb))

            # Create canvas to display the images
            canvas = Canvas(side_by_side_window, width=resized_image.shape[1], height=resized_image.shape[0])
            canvas.pack()
            canvas.create_image(0, 0, image=imgtk_combined, anchor=NW)

            # Calculate the difference for both elbows
            left_angle_diff = abs(self.extension_left_angle - self.flexion_left_angle)
            right_angle_diff = abs(self.extension_right_angle - self.flexion_right_angle)

            # Display the larger change
            if left_angle_diff > right_angle_diff:
                label_text = f'Left Elbow Change: {left_angle_diff:.2f} degrees'
            else:
                label_text = f'Right Elbow Change: {right_angle_diff:.2f} degrees'

            # Display the angle difference below the images
            label = Label(side_by_side_window, text=label_text, font=("Arial", 14))
            label.pack()

            side_by_side_window.mainloop()


# Main script
if __name__ == "__main__":
    root = Tk()
    App(root, "Elbow Angle Detection")
