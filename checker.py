import numpy as np
import cv2

def mask_detection(model, face):
    
    output = model.predict(np.expand_dims(face, axis = 0))

    if(output[:,1] > 0.001):
        name='no mask found'
        return False
    else:
        name='mask found'
        return True
    
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




