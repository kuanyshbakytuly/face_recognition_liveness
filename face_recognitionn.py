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
from utils_local.utils.utils import texting_in_oval, recognize_face

from passive_liveness.model import AntiSpoofPredict
from passive_liveness.detection import passive_liveness

from detection import custom


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Torch uses {}'.format(device))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Init Model of FR")
model = face_recogntion_model()

print("Init Model of Detection")

#import dlib.cuda as cuda
#print(f'Dlib runs on GPU: {cuda.get_num_devices() == 1}')
detector, predictor = face_detector(path_to_model='src/model_needed _zoom.dat')

print("Init Model of Mask")
mask_detection = mask_model(path_to_model='src/mask.h5', device=device)

print("Init Model of Glass")
glass_detection, glass_input, glass_output = glass_model()

path_to_caffemodel = 'passive_liveness/detection_model/Widerface-RetinaFace.caffemodel'
path_to_deploy = 'passive_liveness/detection_model/deploy.prototxt'

path_to_MiniFASNetV2 = 'passive_liveness/anti_spoofing/2.7_80x80_MiniFASNetV2.pth'
path_to_MiniFASNetV1SE = 'passive_liveness/anti_spoofing/4_0_0_80x80_MiniFASNetV1SE.pth'

print("Init of Passive Liveness")
passive_detection = AntiSpoofPredict(0, (path_to_deploy, path_to_caffemodel), (path_to_MiniFASNetV2, path_to_MiniFASNetV1SE))

print("Init of Tailgatin Model")
tailgating_detection = custom(path_or_model='src/crowdhuman_yolov5m.pt', device=device)

print("Init Face Database")
database_path = "face_database.pkl"
if os.path.exists(database_path):
    with open(database_path, 'rb') as db_file:
        face_database = pickle.load(db_file)
else:
    face_database = {}

face_locations = []
status = False
request_for_tail = False
status_tailgating = False
text = ''
sub_text = ''
status_text = ''


# Set parameters for oval
camera_height = 720
camera_width = 1280
hyperp_text = [(camera_width // 2, camera_height // 2), 150]

counter_for_tail = 0
counter_for_status = 0

# 25 frames per second and overall 25 frame/s * 5s = 125 frames
status_time = 25 * 2

# 25 frames per second and overall 25 frame/s * 5s = 125 frames
terminal_closed_time = 25 * 10

print("Starting Video")
def register_from_video(face_camera: int, tailgating_camera: int):
    global face_locations, status, text, person_name, hyperp_text, sub_text, request_for_tail, counter_for_tail
    global terminal_closed_time, status_time, status_text, counter_for_status, status_tailgating
    
    video_capture_face = cv2.VideoCapture(face_camera)
    video_capture_tail = cv2.VideoCapture(tailgating_camera)

    while True:
        ret_face, frame_camera_face = video_capture_face.read()
        frame_camera_face = cv2.flip(frame_camera_face, 1)

        ret_tail, frame_camera_tail = video_capture_tail.read()
        frame_tail = cv2.flip(frame_camera_tail, 1)

        #Face Camera
        if status == False:
            #Face Detection
            frame_face = frame_camera_face.copy()
            face, landmarks, status_face = checker.face_detection(detector, predictor, frame_camera_face, frame_face)

            if status_face == 2:
                #Sending to backend about Tailgating
                status_text = "Camera1 checked Tailgating"
                request_for_tail = True
            
            if status_face:
                frame_camera_face, status_zoom, status_text, landmarks = video_feed(landmarks, frame_camera_face, face)

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

                                    name = recognize_face(model, face, face_database)
                                    status_text = name

                                    status = True

        # Checking possible tailagting from Camera 1 when more than 1 face is detected
        if request_for_tail:
            frame_tail, status_tailgating = checker.taligatin_detection(tailgating_detection, frame_tail)

            if status_tailgating:
                #Sending to Backend about Tailgating
                status_text =  "Double Checked Tailgating"

        # Checking Zone for tailgating after succesfully passing Face Recogntion by 1 person
        if status:
            frame_tail, status_tailgating = checker.taligatin_detection(tailgating_detection, frame_tail)   

            if not status_tailgating:
                #Sending to Backend about Tailgating
                status_text = "Tailgating"

        frame_camera_face = texting_in_oval(frame_camera_face, status_text, sub_text, hyperp_text[0], hyperp_text[1])

        # Display the resulting frame
        cv2.imshow('Camera 1', frame_camera_face)
        cv2.imshow('Camera 2', frame_tail)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('f'):
            status = False
            request_for_tail = False
            status_tailgating = False

    # Release the capture when everything is done
    video_capture_face.release()
    video_capture_tail.release()
    cv2.destroyAllWindows()

# Example usage
register_from_video(1, 0)
