import cv2
import pickle
import os
import numpy as np
from face_landmark import video_feed
import dlib
from face_recognition_model import model_init
from utils_local import image_to_embedding
import glob


print("Init Model of FR")
model = model_init()

# Initialize face detector and shape predictor outside of the video_feed function
print("Init Model of Detection")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model_needed _zoom.dat')

print("Init Face Database")
database_path = "face_database.pkl"
if os.path.exists(database_path):
    with open(database_path, 'rb') as db_file:
        face_database = pickle.load(db_file)
else:
    face_database = {}

def face_detect(image):
    #Face Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    for face in faces:
        # Get the coordinates of the face rectangle
        startX = face.left()
        startY = face.top()
        endX = face.right()
        endY = face.bottom()

        face_locations = [startX, startY, endX, endY]
        
    
        if not face_locations:
            return []

        face = image[startY:endY, startX:endX]
        
        return face

folder = glob.glob("faces/train/*")

for img in folder:
    name = img.split("/")[0]
    image = cv2.imread(img)
    face = face_detect(image)

    if not face:
        print(name+' is not detected')
        continue

    embedding = image_to_embedding(face, model)

    face_database[name] = embedding

    # Save the updated database
    with open(database_path, 'wb') as db_file:
        pickle.dump(face_database, db_file)
    
    print(name + ' is saved')