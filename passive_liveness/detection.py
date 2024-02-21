import numpy as np
from passive_liveness.crop import CropImage

image_cropper = CropImage()

def passive_liveness(model, frame):
    MiniFASNetV2 = model.MiniFASNetV2
    MiniFASNetV1SE = model.MiniFASNetV1SE

    MiniFASNetV2_params = model.MiniFASNetV2_params
    MiniFASNetV1SE_params = model.MiniFASNetV1SE_params

    image_bbox = model.get_bbox(frame)
    prediction = np.zeros((1, 3))

    MiniFASNetV2_params["org_img"] = frame
    MiniFASNetV2_params["bbox"] = image_bbox

    MiniFASNetV1SE_params["org_img"] = frame
    MiniFASNetV1SE_params["bbox"] = image_bbox


    face_for_MiniFASNetV2 = image_cropper.crop(**MiniFASNetV2_params)
    face_for_MiniFASNetV1SE = image_cropper.crop(**MiniFASNetV1SE_params)

    prediction_of_MiniFASNetV2 = model.predict(MiniFASNetV2, face_for_MiniFASNetV2)
    prediction_of_MiniFASNetV1SE = model.predict(MiniFASNetV1SE, face_for_MiniFASNetV1SE)

    prediction = prediction_of_MiniFASNetV2 + prediction_of_MiniFASNetV1SE
    label = np.argmax(prediction)

    return label == 1

