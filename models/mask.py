from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# model.predict(np.expand_dims(img, axis = 0) )

def mask_model(path_to_model):
    num_classes = 2
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(num_classes, activation='softmax'))

    # not using the first layer for training
    model.layers[0].trainable = False

    model.compile(optimizer='sgd', 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
    
    model.load_weights(path_to_model)
    
    return model