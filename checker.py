import numpy as np
import cv2
from utils_local.utils.utils import get_center, calculate_distance


def face_detection(detector, predictor, frame, frame_face):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    #If no face
    if len(faces) == 0:
        return [], [], 0
    
    #Tailgating
    if len(faces) > 1:
        return [], [], 2

    face = faces[0]
    startX = face.left()
    startY = face.top()
    endX = face.right()
    endY = face.bottom()

    #Get landmarks
    landmarks = predictor(gray, face)

    #Get Face from Frame
    face_locations = [startX, startY, endX, endY]
    face = frame_face[startY - 10:endY + 10, startX - 10:endX + 10]
    
    if not face_locations:
        return 'No Face Detected'
    
    return face, landmarks, 1

def mask_detection(model, face):
    
    output = model(np.expand_dims(face, axis = 0))

    if(output[:,1] > 0.001):
        name='no mask found'
        return False
    else:
        name='mask found'
        return True
    
def taligatin_detection(model, frame):
    output = model(frame)

    coordinates = output.pred[0].numpy()

    heads = []
    for bbox in coordinates:
        x1, y1, x2, y2, confidence, class_id = bbox
        confidence*=100 
        if class_id == 1 and int(confidence) > 50:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            # Draw rectangle around the head
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Optionally, add text
            cv2.putText(frame, f'{round(confidence, 1)}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            heads.append(1)

    if len(heads) == 0:
        print("No people")
        return frame, True
    
    if len(heads) == 1:
        print('One people')
        return frame, True
    
    if len(heads) > 1:
        print('There are more than 2 people')
        return frame, False

    '''center1 = get_center(heads[0])
    center2 = get_center(heads[1])

    distance = calculate_distance(center1, center2)

    # Define a threshold for "near"
    distance_threshold = 100  # This value depends on your specific requirements
    if distance < distance_threshold:
        print("Heads are near each other.")
        return frame
    else:
        print("Heads are not near each other.")
        return frame'''

    
def glass_detection(interpreter, input_details, output_details, face):

    image = cv2.resize(face, (160, 160))
    input_data = np.array(image, dtype=np.float32)
    interpreter.set_tensor(input_details, np.expand_dims(input_data, axis=0))

    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details)

    return output_data[0][0] < 0

def eyes_detection(eye):
    return eye < 0.27




