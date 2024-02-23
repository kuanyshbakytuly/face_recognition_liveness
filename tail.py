import cv2
# Start video stream
from detection import custom
from checker import taligatin_detection

print("Init of Tailgatin Model")
tailgating = custom(path_or_model='src/crowdhuman_yolov5m.pt')

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
    
    return frame

    '''if len(heads) == 0:
        print("No people")
        return frame, True
    
    if len(heads) == 1:
        print('One people')
        return frame, True
    
    if len(heads) > 1:
        print('There are more than 2 people')
        return frame, False'''

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

cap = cv2.VideoCapture(0)

counter = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    counter+=1

    if not ret:
        break

    if counter % 10 == 0:
        frame = taligatin_detection(tailgating, frame)



    # Display the frame for testing purposes
    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()


