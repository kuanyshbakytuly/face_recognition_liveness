from models.vgg_face import face_recogntion_model
from models.mask import mask_model
from models.glass import glass_model
from models.face_detector import face_detector

from utils_local.utils.utils import image_to_embedding
import cv2

class Models:

    def __init__(self):
        print("Init Model of Face Recognition")
        self.face_recognition_model = face_recogntion_model()

        print("Init Model of Face Landmark Detection")
        self.face_detecter, self.face_landmark_detectier = face_detector()

        print("Init Model of Mask")
        self.mask_detection = mask_model(path_to_model='src/mask.h5')

        print("Init Model of Glass")
        self.glass_detection = glass_model()

    def face_recognition(self, image):
        return image_to_embedding(self.face_recognition_model, image)
    
    def face_landmark_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_detecter(gray, 0)

        faces = []
        landmarks = []

        for face in detected_faces:
            # Get the coordinates of the face rectangle
            startX = face.left()
            startY = face.top()
            endX = face.right()
            endY = face.bottom()

            landmark = self.face_landmark_detectier(gray, face)
            face = image[startY - 10:endY + 10, startX - 10:endX + 10]

            faces.append(face)
            landmarks.append(landmark)

        return faces, landmarks
    
    







        
    


