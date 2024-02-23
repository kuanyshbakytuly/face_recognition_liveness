import cv2
import pickle
import os
import tensorflow as tf
import torch

from utils_local.models.vgg_face import face_recogntion_model
from utils_local.models.mask import mask_model
from utils_local.models.glass import glass_model
from utils_local.models.face_detector import face_detector
from utils_local.models.eyes import eyes_area_segmentation

import checker
from utils_local.utils.face_landmark import video_feed
from utils_local.utils.utils import texting_in_oval, image_to_embedding

from passive_liveness.model import AntiSpoofPredict
from passive_liveness.detection import passive_liveness


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Torch uses {}'.format(device))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Init Model of FR in TensorFlow")
model = face_recogntion_model()

print("Init Model of Detection in Dlib")
#import dlib.cuda as cuda
#print(f'Dlib runs on GPU: {cuda.get_num_devices() == 1}')
detector, predictor = face_detector(path_to_model='src/model_needed _zoom.dat')

print("Init Model of Mask in Torch")
mask_detection = mask_model(path_to_model='src/mask.h5', device=device)

print("Init Model of Glass in TensorFlow")
glass_detection, glass_input, glass_output = glass_model()

path_to_caffemodel = 'passive_liveness/detection_model/Widerface-RetinaFace.caffemodel'
path_to_deploy = 'passive_liveness/detection_model/deploy.prototxt'

path_to_MiniFASNetV2 = 'passive_liveness/anti_spoofing/2.7_80x80_MiniFASNetV2.pth'
path_to_MiniFASNetV1SE = 'passive_liveness/anti_spoofing/4_0_0_80x80_MiniFASNetV1SE.pth'
print("Init of Passive Liveness")
passive_detection = AntiSpoofPredict(0, (path_to_deploy, path_to_caffemodel), (path_to_MiniFASNetV2, path_to_MiniFASNetV1SE))

print("Init Face Database")
database_path = "face_database.pkl"
if os.path.exists(database_path):
    with open(database_path, 'rb') as db_file:
        face_database = pickle.load(db_file)
else:
    face_database = {}

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

                if status_zoom:
                    #Mask Detection
                    status_mask = checker.mask_detection(mask_detection, face)
                    status_text = 'Mask is detected'

                    if not status_mask:
                        #Glass Detection
                        status_glass = checker.glass_detection(glass_detection, glass_input, glass_output, face)
                        status_text = 'Glass is detected'

                        if not status_glass:
                            #Eyes Detection
                            status_eyes = checker.eyes_detection(eyes_area_segmentation(landmarks=landmarks))
                            status_text = 'Closed Eyes is detected'

                            if not status_eyes:           
                                #Liveness Detection
                                status_liveness = passive_liveness(passive_detection, frame_face)
                                status_text = 'Spoof is detected'

                                if status_liveness:
                                    #Converting image to embdedding
                                    tf.debugging.set_log_device_placement(True)
                                    embedding = image_to_embedding(model, face)
                                    status_text = f"{person_name} has already been registered successfully."

                                    if not person_name in face_database:
                                        face_database[person_name] = embedding

                                        # Save the updated database
                                        with open(database_path, 'wb') as db_file:
                                            pickle.dump(face_database, db_file)

                                        text = f"{person_name} has been registered successfully."
                                        status_text = text
                                    sub_text = "To Continue Registration TAP F, and enter name"
                                    status = True


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
