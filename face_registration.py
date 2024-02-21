import cv2
import pickle
import os

from models.vgg_face import face_recogntion_model
from models.mask import mask_model
from models.glass import glass_model
from models.face_detector import face_detector
from models.eyes import eyes_area_segmentation

import checker
from utils_r.face_landmark import video_feed
from utils_r.utils import texting_in_oval, image_to_embedding

from passive_liveness.model import AntiSpoofPredict
from passive_liveness.detection import passive_liveness


print("Init Model of FR")
model = face_recogntion_model()

print("Init Model of Detection")
detector, predictor = face_detector(path_to_model='src/model_needed _zoom.dat')

print("Init Model of Mask")
mask_detection = mask_model(path_to_model='src/mask.h5')

print("Init Model of Glass")
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
hyperp_text = ((25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

person_name = 'K'


print("Starting Video")
def register_from_video():
    global face_locations, status, text, person_name, hyperp_text
    
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)

        frame_face = frame.copy()

        if status == False:
            #Face Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray, 0)
            for face in faces:
                # Get the coordinates of the face rectangle
                startX = face.left()
                startY = face.top()
                endX = face.right()
                endY = face.bottom()

                face_locations = [startX, startY, endX, endY]
        
    
            if not face_locations:
                continue

            face = frame_face[startY - 10:endY + 10, startX - 10:endX + 10]
            frame, status_zoom, status_text, hyperp_text = video_feed(frame, faces, face, predictor, gray)

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
                        status_eyes = checker.eyes_detection(eyes_area_segmentation(landmarks=hyperp_text[0]))
                        status_text = 'Closed Eyes is detected'

                        if not status_eyes:           
                            #Liveness Detection
                            status_liveness = passive_liveness(passive_detection, frame_face)
                            status_text = 'Spoof is detected'

                            if status_liveness:
                                #Converting image to embdedding

                                #embedding = image_to_embedding(model, face)
                                '''if person_name in face_database:
                                    text = f"{person_name} has been already registered successfully."
                                    status = True
                                    continue
                                else:
                                    face_database[person_name] = embedding

                                    # Save the updated database
                                with open(database_path, 'wb') as db_file:
                                    pickle.dump(face_database, db_file)

                                text = f"{person_name} has been registered successfully."'''
                                status_text = 'Accepted'

        frame = texting_in_oval(frame, status_text, hyperp_text[1], hyperp_text[2])
        status_text = ''
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('f'):
            status = False
            person_name = input("Enter Your Name: ")

    # Release the capture when everything is done
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
register_from_video()
