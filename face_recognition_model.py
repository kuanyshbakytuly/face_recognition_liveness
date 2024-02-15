from keras_vggface.vggface import VGGFace

def model_init():
    # Convolution Features
    vgg_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max
    return vgg_features