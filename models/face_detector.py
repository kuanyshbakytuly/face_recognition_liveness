import dlib

def face_detector(path_to_model):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_model)

    return detector, predictor