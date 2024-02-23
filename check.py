def taligatin_detection(model, frame):
    output = model(frame)

    coordinates = output.pred[0].numpy()

    heads = coordinates[coordinates[:, 5] == 1 and coordinates[:, 4] > 0.80]

    '''for bbox in heads:
        x1, y1, x2, y2, confidence, class_id = map(int, bbox)
        # Draw rectangle around the head
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Optionally, add text
        cv2.putText(frame, 'Head', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)'''

    if len(heads) == 0:
        print("No people")
        return True
    
    if len(heads) == 1:
        print('One people')
        return True
    
    if len(heads) > 1:
        print('There are more than 2 people')
        return False

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
