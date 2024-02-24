import cv2
import pickle
import os

from utils_local.models.face_detector import face_detector

import checker
from utils_local.utils.face_landmark import video_feed
from utils_local.utils.utils import texting_in_oval

print("Init Model of Detection in Dlib")
#import dlib.cuda as cuda
#print(f'Dlib runs on GPU: {cuda.get_num_devices() == 1}')
detector, predictor = face_detector(path_to_model='src/model_needed _zoom.dat')

face_locations = []
status = False
text = ''
sub_text = ''
status_text = ''

# Set parameters for oval
camera_height = 720
camera_width = 1280
hyperp_text = [(camera_width // 2, camera_height // 2), 150]

counter_for_tail = 0
counter_for_status = 0

person_name = 'Kuanysh'

print("Starting Video")
def register_from_video(face_camera: int):
    global face_locations, status, text, person_name, hyperp_text, sub_text, status_text, hyperp_text
    
    video_capture = cv2.VideoCapture(face_camera)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)

        #Face Camera
        if status == False:
            #Face Detection
            frame_face = frame.copy()
            face, landmarks, status_face = checker.face_detection(detector, predictor, frame, frame_face)
            
            if status_face:
                frame, status_zoom, status_text, landmarks = video_feed(landmarks, frame, face)


        if cv2.waitKey(1) & 0xFF == ord('f'):
            status = False
            status_text = ""
            sub_text = ""
            frame = texting_in_oval(frame, status_text, sub_text, hyperp_text[0], hyperp_text[1])
            cv2.imshow('Video', frame)
            person_name = input("Enter Your Name: ")

        frame = texting_in_oval(frame, status_text, sub_text, hyperp_text[0], hyperp_text[1])

        if not status:
            status_text = ""

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when everything is done
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
register_from_video(1)
