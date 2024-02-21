import cv2
import pickle
import os
from utils.face_landmark import video_feed
import dlib
from utils.utils import recognize_face
from models.vgg_face import model_init


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

# Initialize some variables
face_encodings = []
face_names = []
face_locations = []
name = "Unknown"
status = False

print("Starting Video")
def recognition_from_video():
    global face_database, face_locations, name, status
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if cv2.waitKey(1) & 0xFF == ord('f'):
            name = 'Unknown'
            status = False
            face_locations = []
            continue

        frame = cv2.flip(frame, 1)

        #Face Detection
        if status == False:
            c = 0
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
                name = recognize_face(model, face, face_database)
                status = True

                if not name:
                    name = 'Unknown'
        
        coordinates = (25, 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(frame, name, coordinates, font,
                        fontScale, color, thickness, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Video', frame)


        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when everything is done
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
recognition_from_video()
