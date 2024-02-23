import cv2
import numpy as np
import math

def get_center(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(center1, center2):
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def texting_in_oval(image, text, sub_text, oval_center, axes):
    if text == "":
        return image
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = oval_center[0] - text_size[0] // 2
    text_y = oval_center[1] - axes - 10 # 10 pixels above the top of the oval 

    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   
    if sub_text != '':
        text_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = oval_center[0] - text_size[0] // 2
        text_y += 40
        cv2.putText(image, sub_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
    return image

def check_image_quality(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check for blur
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 50:  # The threshold for fm might need adjustment based on your requirements
        return (False, "Image is blurry.")
    else:
        return (True, "Image is not blurry.")
    
    # Assess lighting
    brightness = np.mean(gray)
    if brightness < 50 or brightness > 200:  # These thresholds are arbitrary and should be adjusted
        return (False, "Poor lighting conditions.")
    else:
        return (True, "Good lighting conditions.")
    
def image_to_embedding(model, image):
    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA) 
    image = cv2.resize(image, (224, 224)) 
    embedding = model(np.expand_dims(image, axis=0))
    return embedding

def cosine_similarity(A, B):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(A, np.transpose(B))
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def recognize_face(model, face_image, input_embeddings):

    embedding = image_to_embedding(model, face_image)
    
    minimum_distance = -11
    name = None
    
    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():
        
        euclidean_distance = cosine_similarity(embedding, input_embedding)
        
        if euclidean_distance > minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
    
    if minimum_distance > 0.6:
        return str(name)
    else:
        return 'Unknown'