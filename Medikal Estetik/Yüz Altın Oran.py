import cv2
import mediapipe as mp
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

bottom_nose_idx = 2
top_lip_idx = 13
bottom_lip_idx = 14
chin_idx = 152
left_eye_inner_idx = 133
right_eye_inner_idx = 362
glabella_idx = 168
top_forehead_idx = 10
leftmost_face_idx = 234
rightmost_face_idx = 454
nose_tip_idx = 1
chin_tip_idx = 152
left_inner_eyebrow_idx = 55
right_inner_eyebrow_idx = 285
left_eye_center_idx = 468
right_eye_center_idx = 473
left_eye_outer_idx = 33
right_eye_outer_idx = 263
left_mouth_corner_idx = 61
right_mouth_corner_idx = 291
left_nose_wing_idx = 49
right_nose_wing_idx = 279

def setup_ratio_points(face_landmarks):
    global ratio_points
    ratio_points = {
        'top_forehead': calculate_highest_point_with_percentage_offset(face_landmarks, height, width),

        'eyebrow_midpoint': (
            int((face_landmarks.landmark[55].x + face_landmarks.landmark[285].x) * width / 2),
            int((face_landmarks.landmark[55].y + face_landmarks.landmark[285].y) * height / 2)
        ),

        'nose_tip': (
            int(face_landmarks.landmark[1].x * width),
            int(face_landmarks.landmark[1].y * height)
        ),
        'chin_tip': (
            int(face_landmarks.landmark[152].x * width),
            int(face_landmarks.landmark[152].y * height)
        ),
    }

def calculate_highest_point_with_percentage_offset(face_landmarks, img_height, img_width):
    top_forehead = face_landmarks.landmark[top_forehead_idx]
    chin = face_landmarks.landmark[chin_idx]

    top_forehead_x = int(top_forehead.x * img_width)
    top_forehead_y = int(top_forehead.y * img_height)
    chin_y = int(chin.y * img_height)

    face_height = chin_y - top_forehead_y
    offset_in_pixels = int(face_height * 0.1)  # 10% offset

    highest_y = max(0, top_forehead_y - offset_in_pixels)
    return top_forehead_x, highest_y

def load_image(image_path):
    global loaded_image, width, height
    loaded_image = cv2.imread(image_path)
    height, width, _ = loaded_image.shape

def show_nose_lips_chin_ratio():
    global loaded_image
    if loaded_image is None:
        return

    image = loaded_image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            bottom_nose = face_landmarks.landmark[bottom_nose_idx]
            top_lip = face_landmarks.landmark[top_lip_idx]
            bottom_lip = face_landmarks.landmark[bottom_lip_idx]
            chin = face_landmarks.landmark[chin_idx]

            bottom_nose_x, bottom_nose_y = int(bottom_nose.x * width), int(bottom_nose.y * height)
            top_lip_x, top_lip_y = int(top_lip.x * width), int(top_lip.y * height)
            bottom_lip_x, bottom_lip_y = int(bottom_lip.x * width), int(bottom_lip.y * height)
            chin_x, chin_y = int(chin.x * width), int(chin.y * height)

            bottom_lip_to_chin_dist = math.sqrt((bottom_lip_x - chin_x) ** 2 + (bottom_lip_y - chin_y) ** 2)
            nose_to_top_lip_dist = math.sqrt((bottom_nose_x - top_lip_x) ** 2 + (bottom_nose_y - top_lip_y) ** 2)
            nose_lips_chin_ratio = bottom_lip_to_chin_dist / nose_to_top_lip_dist if nose_to_top_lip_dist != 0 else 0

            cv2.line(image, (bottom_nose_x, bottom_nose_y), (top_lip_x, top_lip_y), (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(image, (bottom_lip_x, bottom_lip_y), (chin_x, chin_y), (255, 0, 0), 3, cv2.LINE_AA)

            cv2.putText(image, f"Nose-Lips-Chin Ratio: {nose_lips_chin_ratio:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1.618", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)


            display_image(image)

def show_eye_symmetry_ratio():
    global loaded_image
    if loaded_image is None:
        return

    image = loaded_image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_inner = face_landmarks.landmark[left_eye_inner_idx]
            right_eye_inner = face_landmarks.landmark[right_eye_inner_idx]
            glabella = face_landmarks.landmark[glabella_idx]

            left_eye_x, left_eye_y = int(left_eye_inner.x * width), int(left_eye_inner.y * height)
            right_eye_x, right_eye_y = int(right_eye_inner.x * width), int(right_eye_inner.y * height)
            glabella_x, glabella_y = int(glabella.x * width), int(glabella.y * height)

            left_eye_to_glabella_dist = math.sqrt((left_eye_x - glabella_x) ** 2 + (left_eye_y - glabella_y) ** 2)
            right_eye_to_glabella_dist = math.sqrt((right_eye_x - glabella_x) ** 2 + (right_eye_y - glabella_y) ** 2)
            eye_symmetry_ratio = left_eye_to_glabella_dist / right_eye_to_glabella_dist if right_eye_to_glabella_dist != 0 else 0

            cv2.line(image, (left_eye_x, left_eye_y), (glabella_x, glabella_y), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.line(image, (right_eye_x, right_eye_y), (glabella_x, glabella_y), (0, 255, 0), 3, cv2.LINE_AA)

            cv2.putText(image, f"Eye Symmetry: {eye_symmetry_ratio:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


            display_image(image)

def show_both_ratios():
    global loaded_image
    if loaded_image is None:
        return

    image = loaded_image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            bottom_nose = face_landmarks.landmark[bottom_nose_idx]
            top_lip = face_landmarks.landmark[top_lip_idx]
            bottom_lip = face_landmarks.landmark[bottom_lip_idx]
            chin = face_landmarks.landmark[chin_idx]
            left_eye_inner = face_landmarks.landmark[left_eye_inner_idx]
            right_eye_inner = face_landmarks.landmark[right_eye_inner_idx]
            glabella = face_landmarks.landmark[glabella_idx]

            bottom_nose_x, bottom_nose_y = int(bottom_nose.x * width), int(bottom_nose.y * height)
            top_lip_x, top_lip_y = int(top_lip.x * width), int(top_lip.y * height)
            bottom_lip_x, bottom_lip_y = int(bottom_lip.x * width), int(bottom_lip.y * height)
            chin_x, chin_y = int(chin.x * width), int(chin.y * height)
            left_eye_x, left_eye_y = int(left_eye_inner.x * width), int(left_eye_inner.y * height)
            right_eye_x, right_eye_y = int(right_eye_inner.x * width), int(right_eye_inner.y * height)
            glabella_x, glabella_y = int(glabella.x * width), int(glabella.y * height)

            bottom_lip_to_chin_dist = math.sqrt((bottom_lip_x - chin_x) ** 2 + (bottom_lip_y - chin_y) ** 2)
            nose_to_top_lip_dist = math.sqrt((bottom_nose_x - top_lip_x) ** 2 + (bottom_nose_y - top_lip_y) ** 2)
            nose_lips_chin_ratio = bottom_lip_to_chin_dist / nose_to_top_lip_dist if nose_to_top_lip_dist != 0 else 0

            left_eye_to_glabella_dist = math.sqrt((left_eye_x - glabella_x) ** 2 + (left_eye_y - glabella_y) ** 2)
            right_eye_to_glabella_dist = math.sqrt((right_eye_x - glabella_x) ** 2 + (right_eye_y - glabella_y) ** 2)
            eye_symmetry_ratio = left_eye_to_glabella_dist / right_eye_to_glabella_dist if right_eye_to_glabella_dist != 0 else 0

            cv2.line(image, (bottom_nose_x, bottom_nose_y), (top_lip_x, top_lip_y), (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(image, (bottom_lip_x, bottom_lip_y), (chin_x, chin_y), (255, 0, 0), 3, cv2.LINE_AA)

            cv2.line(image, (left_eye_x, left_eye_y), (glabella_x, glabella_y), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.line(image, (right_eye_x, right_eye_y), (glabella_x, glabella_y), (0, 255, 0), 3, cv2.LINE_AA)

            cv2.putText(image, f"Nose-Lips-Chin Ratio: {nose_lips_chin_ratio:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1.618", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, f"Eye Symmetry: {eye_symmetry_ratio:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1", (20, 120),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


            display_image(image)

def show_face_height_to_width_ratio():
    global loaded_image
    if loaded_image is None:
        return

    image = loaded_image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            highest_x, highest_y = calculate_highest_point_with_percentage_offset(face_landmarks, height, width)

            chin = face_landmarks.landmark[chin_idx]
            lowest_x, lowest_y = int(chin.x * width), int(chin.y * height)

            leftmost_face = face_landmarks.landmark[leftmost_face_idx]
            rightmost_face = face_landmarks.landmark[rightmost_face_idx]
            leftmost_x, leftmost_y = int(leftmost_face.x * width), int(leftmost_face.y * height)
            rightmost_x, rightmost_y = int(rightmost_face.x * width), int(rightmost_face.y * height)

            face_height = math.sqrt((highest_x - lowest_x) ** 2 + (highest_y - lowest_y) ** 2)
            face_width = math.sqrt((leftmost_x - rightmost_x) ** 2 + (leftmost_y - rightmost_y) ** 2)
            height_to_width_ratio = face_height / face_width if face_width != 0 else 0

            cv2.line(image, (highest_x, highest_y), (lowest_x, lowest_y), (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(image, (leftmost_x, leftmost_y), (rightmost_x, rightmost_y), (0, 255, 255), 3, cv2.LINE_AA)

            cv2.putText(image, f"Face Height-to-Width Ratio: {height_to_width_ratio:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1.618", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


            display_image(image)

def show_cd_ratio():
    global loaded_image
    if loaded_image is None:
        return

    image = loaded_image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            highest_x, highest_y = calculate_highest_point_with_percentage_offset(face_landmarks, height, width)

            nose_tip = face_landmarks.landmark[nose_tip_idx]
            chin_tip = face_landmarks.landmark[chin_tip_idx]
            nose_x, nose_y = int(nose_tip.x * width), int(nose_tip.y * height)
            chin_x, chin_y = int(chin_tip.x * width), int(chin_tip.y * height)

            C = math.sqrt((nose_x - chin_x) ** 2 + (nose_y - chin_y) ** 2)

            left_eyebrow = face_landmarks.landmark[left_inner_eyebrow_idx]
            right_eyebrow = face_landmarks.landmark[right_inner_eyebrow_idx]
            left_eyebrow_x, left_eyebrow_y = int(left_eyebrow.x * width), int(left_eyebrow.y * height)
            right_eyebrow_x, right_eyebrow_y = int(right_eyebrow.x * width), int(right_eyebrow.y * height)

            midpoint_x = (left_eyebrow_x + right_eyebrow_x) // 2
            midpoint_y = (left_eyebrow_y + right_eyebrow_y) // 2

            D = math.sqrt((highest_x - midpoint_x) ** 2 + (highest_y - midpoint_y) ** 2)

            cd_ratio = C / D if D != 0 else 0

            cv2.line(image, (nose_x, nose_y), (chin_x, chin_y), (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(image, (highest_x, highest_y), (midpoint_x, midpoint_y), (0, 255, 0), 3, cv2.LINE_AA)

            cv2.putText(image, f"Nose - Chin / Forehead Ratio: {cd_ratio:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1.618", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)


            display_image(image)

def show_eye_pupil_to_width_ratio():
    global loaded_image
    if loaded_image is None:
        return

    image = loaded_image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_center = face_landmarks.landmark[left_eye_center_idx]
            right_eye_center = face_landmarks.landmark[right_eye_center_idx]
            left_eye_inner = face_landmarks.landmark[left_eye_inner_idx]
            left_eye_outer = face_landmarks.landmark[left_eye_outer_idx]
            right_eye_inner = face_landmarks.landmark[right_eye_inner_idx]
            right_eye_outer = face_landmarks.landmark[right_eye_outer_idx]

            left_eye_center_x, left_eye_center_y = int(left_eye_center.x * width), int(left_eye_center.y * height)
            right_eye_center_x, right_eye_center_y = int(right_eye_center.x * width), int(right_eye_center.y * height)
            left_eye_inner_x, left_eye_inner_y = int(left_eye_inner.x * width), int(left_eye_inner.y * height)
            left_eye_outer_x, left_eye_outer_y = int(left_eye_outer.x * width), int(left_eye_outer.y * height)
            right_eye_inner_x, right_eye_inner_y = int(right_eye_inner.x * width), int(right_eye_inner.y * height)
            right_eye_outer_x, right_eye_outer_y = int(right_eye_outer.x * width), int(right_eye_outer.y * height)

            left_eye_pupil_distance = math.sqrt((left_eye_center_x - right_eye_center_x) ** 2 + (left_eye_center_y - right_eye_center_y) ** 2)
            left_eye_width_distance = math.sqrt((left_eye_inner_x - left_eye_outer_x) ** 2 + (left_eye_inner_y - left_eye_outer_y) ** 2)
            left_eye_pupil_to_width_ratio = left_eye_pupil_distance / left_eye_width_distance if left_eye_width_distance != 0 else 0

            right_eye_pupil_distance = math.sqrt((left_eye_center_x - right_eye_center_x) ** 2 + (left_eye_center_y - right_eye_center_y) ** 2)
            right_eye_width_distance = math.sqrt((right_eye_inner_x - right_eye_outer_x) ** 2 + (right_eye_inner_y - right_eye_outer_y) ** 2)
            right_eye_pupil_to_width_ratio = right_eye_pupil_distance / right_eye_width_distance if right_eye_width_distance != 0 else 0

            cv2.line(image, (left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y), (255, 0, 0), 3, cv2.LINE_AA)

            cv2.line(image, (left_eye_inner_x, left_eye_inner_y), (left_eye_outer_x, left_eye_outer_y), (0, 255, 0), 3, cv2.LINE_AA)

            cv2.line(image, (right_eye_inner_x, right_eye_inner_y), (right_eye_outer_x, right_eye_outer_y), (0, 0, 255), 3, cv2.LINE_AA)

            cv2.putText(image, f"Right Eye Pupil to Width Ratio: {left_eye_pupil_to_width_ratio:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 2", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Left Eye Pupil to Width Ratio: {right_eye_pupil_to_width_ratio:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 2", (20, 120),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)



            display_image(image)

def show_mouth_nose_ratio():
    global loaded_image
    if loaded_image is None:
        return

    image = loaded_image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_mouth_corner = face_landmarks.landmark[left_mouth_corner_idx]
            right_mouth_corner = face_landmarks.landmark[right_mouth_corner_idx]
            left_nose_wing = face_landmarks.landmark[left_nose_wing_idx]
            right_nose_wing = face_landmarks.landmark[right_nose_wing_idx]

            left_mouth_x, left_mouth_y = int(left_mouth_corner.x * width), int(left_mouth_corner.y * height)
            right_mouth_x, right_mouth_y = int(right_mouth_corner.x * width), int(right_mouth_corner.y * height)

            left_nose_x, left_nose_y = int(left_nose_wing.x * width), int(left_nose_wing.y * height)
            right_nose_x, right_nose_y = int(right_nose_wing.x * width), int(right_nose_wing.y * height)

            mouth_width = math.sqrt((right_mouth_x - left_mouth_x) ** 2 + (right_mouth_y - left_mouth_y) ** 2)
            nose_width = math.sqrt((right_nose_x - left_nose_x) ** 2 + (right_nose_y - left_nose_y) ** 2)

            mouth_nose_ratio = mouth_width / nose_width if nose_width != 0 else 0

            cv2.line(image, (left_mouth_x, left_mouth_y), (right_mouth_x, right_mouth_y), (0, 255, 0), 3, cv2.LINE_AA)  # Green for mouth width
            cv2.line(image, (left_nose_x, left_nose_y), (right_nose_x, right_nose_y), (0, 165, 255), 3, cv2.LINE_AA)  # Orange for nose width

            cv2.putText(image, f"Mouth / Nose Ratio: {mouth_nose_ratio:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1.618", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


            display_image(image)


def show_face_sections(image):
    global ratio_points

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            setup_ratio_points(face_landmarks)  # Initialize ratio_points with landmarks

            section_1_length = math.sqrt((ratio_points['top_forehead'][0] - ratio_points['eyebrow_midpoint'][0]) ** 2 +
                                         (ratio_points['top_forehead'][1] - ratio_points['eyebrow_midpoint'][1]) ** 2)
            section_2_length = math.sqrt((ratio_points['eyebrow_midpoint'][0] - ratio_points['nose_tip'][0]) ** 2 +
                                         (ratio_points['eyebrow_midpoint'][1] - ratio_points['nose_tip'][1]) ** 2)
            section_3_length = math.sqrt((ratio_points['nose_tip'][0] - ratio_points['chin_tip'][0]) ** 2 +
                                         (ratio_points['nose_tip'][1] - ratio_points['chin_tip'][1]) ** 2)

            cv2.line(image, ratio_points['top_forehead'], ratio_points['eyebrow_midpoint'], (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(image, ratio_points['eyebrow_midpoint'], ratio_points['nose_tip'], (0, 255, 0), 3, cv2.LINE_AA)
            cv2.line(image, ratio_points['nose_tip'], ratio_points['chin_tip'], (0, 0, 255), 3, cv2.LINE_AA)

            cv2.putText(image, f"Ust Yuz: {section_1_length:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Orta Yuz: {section_2_length:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Alt Yuz: {section_3_length:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Normal Ratio: 1:1:1", (20, 120),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


            display_image(image)

def show_jawline_to_face_width_ratio():
        global loaded_image
        if loaded_image is None:
            return

        image = loaded_image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                jawline_indices = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152, 377, 400, 379, 365, 397, 288,
                                   361, 454]
                jaw_points = [
                    (int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)) for idx
                    in jawline_indices]

                jawline_width = sum(
                    math.sqrt(
                        (jaw_points[i][0] - jaw_points[i + 1][0]) ** 2 + (jaw_points[i][1] - jaw_points[i + 1][1]) ** 2)
                    for i in range(len(jaw_points) - 1)
                )

                left_jaw_x, left_jaw_y = int(face_landmarks.landmark[234].x * width), int(
                    face_landmarks.landmark[234].y * height)
                right_jaw_x, right_jaw_y = int(face_landmarks.landmark[454].x * width), int(
                    face_landmarks.landmark[454].y * height)
                face_width = math.sqrt((left_jaw_x - right_jaw_x) ** 2 + (left_jaw_y - right_jaw_y) ** 2)

                jawline_to_face_width_ratio = jawline_width / face_width if face_width != 0 else 0

                for i in range(len(jaw_points) - 1):
                    cv2.line(image, jaw_points[i], jaw_points[i + 1], (0, 255, 255), 2, cv2.LINE_AA)

                cv2.line(image, (left_jaw_x, left_jaw_y), (right_jaw_x, right_jaw_y), (255, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(image, f"Jawline-to-Face Width Ratio: {jawline_to_face_width_ratio:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Normal Ratio: ~2", (20, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                display_image(image)
def calculate_nose_lips_chin_ratio():
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            bottom_nose = face_landmarks.landmark[bottom_nose_idx]
            top_lip = face_landmarks.landmark[top_lip_idx]
            bottom_lip = face_landmarks.landmark[bottom_lip_idx]
            chin = face_landmarks.landmark[chin_idx]

            bottom_lip_to_chin_dist = math.sqrt((bottom_lip.x - chin.x) ** 2 + (bottom_lip.y - chin.y) ** 2)
            nose_to_top_lip_dist = math.sqrt((bottom_nose.x - top_lip.x) ** 2 + (bottom_nose.y - top_lip.y) ** 2)
            return bottom_lip_to_chin_dist / nose_to_top_lip_dist if nose_to_top_lip_dist != 0 else 0
    return 0

def calculate_eye_symmetry_ratio():
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_inner = face_landmarks.landmark[left_eye_inner_idx]
            right_eye_inner = face_landmarks.landmark[right_eye_inner_idx]
            glabella = face_landmarks.landmark[glabella_idx]

            left_eye_to_glabella_dist = math.sqrt((left_eye_inner.x - glabella.x) ** 2 + (left_eye_inner.y - glabella.y) ** 2)
            right_eye_to_glabella_dist = math.sqrt((right_eye_inner.x - glabella.x) ** 2 + (right_eye_inner.y - glabella.y) ** 2)
            return left_eye_to_glabella_dist / right_eye_to_glabella_dist if right_eye_to_glabella_dist != 0 else 0
    return 0

def calculate_face_height_to_width_ratio():
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            highest_x, highest_y = calculate_highest_point_with_percentage_offset(face_landmarks, height, width)

            chin = face_landmarks.landmark[chin_idx]
            leftmost_face = face_landmarks.landmark[leftmost_face_idx]
            rightmost_face = face_landmarks.landmark[rightmost_face_idx]

            face_height = math.sqrt((highest_x - chin.x * width) ** 2 + (highest_y - chin.y * height) ** 2)
            face_width = math.sqrt((leftmost_face.x * width - rightmost_face.x * width) ** 2 + (leftmost_face.y * height - rightmost_face.y * height) ** 2)
            return face_height / face_width if face_width != 0 else 0
    return 0

def calculate_cd_ratio():
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            highest_x, highest_y = calculate_highest_point_with_percentage_offset(face_landmarks, height, width)

            nose_tip = face_landmarks.landmark[nose_tip_idx]
            chin_tip = face_landmarks.landmark[chin_tip_idx]
            left_eyebrow = face_landmarks.landmark[left_inner_eyebrow_idx]
            right_eyebrow = face_landmarks.landmark[right_inner_eyebrow_idx]

            C = math.sqrt((nose_tip.x * width - chin_tip.x * width) ** 2 + (nose_tip.y * height - chin_tip.y * height) ** 2)
            midpoint_x = (left_eyebrow.x * width + right_eyebrow.x * width) / 2
            midpoint_y = (left_eyebrow.y * height + right_eyebrow.y * height) / 2
            D = math.sqrt((highest_x - midpoint_x) ** 2 + (highest_y - midpoint_y) ** 2)
            return C / D if D != 0 else 0
    return 0

def calculate_eye_pupil_to_width_ratio():
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_center = face_landmarks.landmark[left_eye_center_idx]
            right_eye_center = face_landmarks.landmark[right_eye_center_idx]
            left_eye_inner = face_landmarks.landmark[left_eye_inner_idx]
            left_eye_outer = face_landmarks.landmark[left_eye_outer_idx]
            right_eye_inner = face_landmarks.landmark[right_eye_inner_idx]
            right_eye_outer = face_landmarks.landmark[right_eye_outer_idx]

            eye_distance = math.sqrt((left_eye_center.x - right_eye_center.x) ** 2 + (left_eye_center.y - right_eye_center.y) ** 2)
            left_eye_width = math.sqrt((left_eye_inner.x - left_eye_outer.x) ** 2 + (left_eye_inner.y - left_eye_outer.y) ** 2)
            right_eye_width = math.sqrt((right_eye_inner.x - right_eye_outer.x) ** 2 + (right_eye_inner.y - right_eye_outer.y) ** 2)
            return eye_distance / (left_eye_width + right_eye_width) if (left_eye_width + right_eye_width) != 0 else 0
    return 0

def calculate_mouth_nose_ratio():
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_mouth_corner = face_landmarks.landmark[left_mouth_corner_idx]
            right_mouth_corner = face_landmarks.landmark[right_mouth_corner_idx]
            left_nose_wing = face_landmarks.landmark[left_nose_wing_idx]
            right_nose_wing = face_landmarks.landmark[right_nose_wing_idx]

            mouth_width = math.sqrt((left_mouth_corner.x - right_mouth_corner.x) ** 2 + (left_mouth_corner.y - right_mouth_corner.y) ** 2)
            nose_width = math.sqrt((left_nose_wing.x - right_nose_wing.x) ** 2 + (left_nose_wing.y - right_nose_wing.y) ** 2)
            return mouth_width / nose_width if nose_width != 0 else 0
    return 0

def calculate_jawline_to_face_width_ratio():
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            jawline_indices = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152, 377, 400, 379, 365, 397, 288, 361, 454]
            jaw_points = [(int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)) for idx in jawline_indices]

            jawline_width = sum(math.sqrt((jaw_points[i][0] - jaw_points[i + 1][0]) ** 2 + (jaw_points[i][1] - jaw_points[i + 1][1]) ** 2) for i in range(len(jaw_points) - 1))
            face_width = math.sqrt((jaw_points[0][0] - jaw_points[-1][0]) ** 2 + (jaw_points[0][1] - jaw_points[-1][1]) ** 2)
            return jawline_width / face_width if face_width != 0 else 0
    return 0

IDEAL_VALUES = {
    "Burun-Dudak-Çene Oranı": 1.618,
    "Göz Simetrisi Oranı": 1.0,
    "Yüz Yüksekliği-Genişliği Oranı": 1.618,
    "Burun-Çene/Alın Oranı": 1.618,
    "Göz Bebeği-Genişlik Oranı": 1.0,
    "Ağız/Burun Oranı": 1.618,
    "Çene Hattı-Yüz Genişliği Oranı": 2.0
}


def analyze_ratios():
    if loaded_image is None:
        result_text.set("Lütfen önce bir resim yükleyin.")
        return

    ratios = {
        "Burun-Dudak-Çene Oranı": calculate_nose_lips_chin_ratio(),
        "Göz Simetrisi Oranı": calculate_eye_symmetry_ratio(),
        "Yüz Yüksekliği-Genişliği Oranı": calculate_face_height_to_width_ratio(),
        "Burun-Çene/Alın Oranı": calculate_cd_ratio(),
        "Göz Bebeği-Genişlik Oranı": calculate_eye_pupil_to_width_ratio(),
        "Ağız/Burun Oranı": calculate_mouth_nose_ratio(),
        "Çene Hattı-Yüz Genişliği Oranı": calculate_jawline_to_face_width_ratio()
    }

    deviations = []  # List to store each deviation percentage for averaging
    result_output = "Yüz Simetrisi Analiz Raporu\n"
    result_output += "=" * 35 + "\n\n"

    for ratio_name, actual_value in ratios.items():
        ideal_value = IDEAL_VALUES[ratio_name]
        deviation = abs((actual_value - ideal_value) / ideal_value) * 100
        deviations.append(deviation)

        result_output += f"{ratio_name}:\n"
        result_output += f"  - Gerçek Değer: {actual_value:.2f}\n"
        result_output += f"  - İdeal Değer: {ideal_value}\n"
        result_output += f"  - Sapma: {deviation:.2f}%\n\n"

    average_deviation = sum(deviations) / len(deviations)
    golden_ratio_score = 100 - average_deviation

    result_output += "=" * 35 + "\n"
    result_output += f"Altın Oran Skoru: {golden_ratio_score:.2f}%\n"
    result_output += "=" * 35

    result_text.set(result_output)
    print("selam")




def display_image(image):
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image \
        = Image.fromarray(processed_image)
    tk_image = ImageTk.PhotoImage(pil_image)

    image_canvas.create_image(0, 0, anchor="nw", image=tk_image)
    image_canvas.image = tk_image
    image_canvas.config(scrollregion=image_canvas.bbox("all"))

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        load_image(file_path)
        display_image(loaded_image)

root = tk.Tk()
root.title("Facial Symmetry Ratio Calculator")

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

nose_lips_chin_button = tk.Button(root, text="Show Nose-Lips-Chin Ratio", command=show_nose_lips_chin_ratio)
nose_lips_chin_button.pack()

eye_symmetry_button = tk.Button(root, text="Show Eye Symmetry Ratio", command=show_eye_symmetry_ratio)
eye_symmetry_button.pack()

both_ratios_button = tk.Button(root, text="Show Both Ratios", command=show_both_ratios)
both_ratios_button.pack()

face_height_to_width_button = tk.Button(root, text="Show Face Height-to-Width Ratio", command=show_face_height_to_width_ratio)
face_height_to_width_button.pack()

cd_ratio_button = tk.Button(root, text="Show Nose - Chin / Forehead Ratio", command=show_cd_ratio)
cd_ratio_button.pack()

eye_pupil_to_width_button = tk.Button(root, text="Show Eye Pupil to Width Ratios", command=show_eye_pupil_to_width_ratio)
eye_pupil_to_width_button.pack()

mouth_nose_button = tk.Button(root, text="Show Mouth / Nose Ratio", command=show_mouth_nose_ratio)
mouth_nose_button.pack()

face_sections_button = tk.Button(root, text="Show Face Sections", command=lambda: show_face_sections(loaded_image.copy()))
face_sections_button.pack()

jawline_to_face_width_button = tk.Button(root, text="Show Jawline-to-Face Width Ratio", command=show_jawline_to_face_width_ratio)
jawline_to_face_width_button.pack()

result_text = tk.StringVar()

analyze_button = tk.Button(root, text="Analyze Ratios", command=analyze_ratios)
analyze_button.pack()

result_label = tk.Label(root, textvariable=result_text, justify="left", font=("Helvetica", 10))
result_label.pack()


canvas_frame = tk.Frame(root)
canvas_frame.pack(fill="both", expand=True)

image_canvas = tk.Canvas(canvas_frame, width=600, height=600)
image_canvas.pack(side="left", fill="both", expand=True)

scroll_x = tk.Scrollbar(canvas_frame, orient="horizontal", command=image_canvas.xview)
scroll_x.pack(side="bottom", fill="x")
scroll_y = tk.Scrollbar(canvas_frame, orient="vertical", command=image_canvas.yview)
scroll_y.pack(side="right", fill="y")

image_canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)

root.mainloop()
