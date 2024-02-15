import cv2
import pickle
import os
from face_landmark import video_feed
import dlib
from face_recognition_model import model_init
from utils_local import image_to_embedding


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

face_locations = []
status = False
text = ''

person_name = input("Enter Your Name: ")

print("Starting Video")
def register_from_video():
    global face_locations, status, text, person_name
    
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)

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

            face = frame[startY - 10:endY + 10, startX - 10:endX + 10]
            frame, status_zoom = video_feed(frame, faces, face, predictor, gray)

            if status_zoom:
                embedding = image_to_embedding(model, face)
                
                if person_name in face_database:
                    text = f"{person_name} has been already registered successfully."
                    status = True
                    continue
                else:
                    face_database[person_name] = embedding

                    # Save the updated database
                with open(database_path, 'wb') as db_file:
                    pickle.dump(face_database, db_file)

                text = f"{person_name} has been registered successfully."

        coordinates = (25, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(frame, text, coordinates, font,
                        fontScale, color, thickness, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('f'):
            status = False
            print(text)
            person_name = input("Enter Your Name: ")

    # Release the capture when everything is done
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
register_from_video()
