import cv2
import numpy as np

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
        return None