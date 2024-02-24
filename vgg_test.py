import cv2
import pickle
import os
import tensorflow as tf

from utils_local.models.vgg_face import face_recogntion_model

from utils_local.utils.utils import texting_in_oval, image_to_embedding


print("Init Model of FR in TensorFlow")
model = face_recogntion_model()

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
        embedding = image_to_embedding(model, frame)
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
