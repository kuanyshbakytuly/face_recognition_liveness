import cv2
import numpy as np
from imutils import face_utils
from utils_r.utils import check_image_quality, texting_in_oval
from models.eyes import eye_aspect_ratio

# Camera resolution
camera_height = 720
camera_width = 1280

# Set parameters for oval
width = 300
height = 400
oval_center = (camera_width // 2, camera_height // 2) 
inner_center = oval_center
axes = (width//2, height//2)
inner_axes = (int(width//3), int(height//3))
status_text = "Please zoom in"
status_quality = False
status = False
landmarks = []


def video_feed(frame, faces, facee, predictor, gray):
    global width, height, status_text, oval_center, inner_center, axes, inner_axes, status, status_quality, landmarks

    face_in_oval = False

    for face in faces:
        # Get the facial landmarks for the current face
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        landmark_points = landmarks[:3]

        for (x, y) in landmark_points:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        distances_inner = ((landmark_points[:, 0]-inner_center[0])/inner_axes[0])**2 + (
            (landmark_points[:, 1]-inner_center[1])/inner_axes[1])**2

        distances_outer = ((landmark_points[:, 0]-oval_center[0])/axes[0])**2 + (
            (landmark_points[:, 1]-oval_center[1])/axes[1])**2

        if np.all(distances_inner >= 1) and np.all(distances_outer <= 1):
            face_in_oval = True
    
    if face_in_oval:
        cv2.ellipse(frame, oval_center, axes,
                    0, 0, 360, (0, 255, 0), 2)
            
        status_quality = check_image_quality(facee)

        if status_quality[0] == False:
            status_text = f"Repeat. {status_quality[1]}"
            status = False

        status = True
        status_text = ''

    else:
        cv2.ellipse(frame, oval_center, axes,
                    0, 0, 360, (255, 0, 0), 2)
        status_text = "Please zoom in"
        status = False

    return frame, status, status_text, [landmarks, oval_center, axes[1]]