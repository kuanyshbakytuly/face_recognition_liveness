import tensorflow as tf

def glass_model():
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="src/glass.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details[0]['index'], output_details[0]['index'], 