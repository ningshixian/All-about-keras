import keras
from keras.layers import Lambda
import keras.backend as K
from sklearn.utils import shuffle
import skimage.io as io
import numpy as np

from utils import data_augmentation

def inception(EMB_VECTOR,IMG_SIZE, use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)

    # # add new dense layer for our labels
    new_output = keras.layers.Dense(EMB_VECTOR,activation='sigmoid')(new_output)
    new_output = Lambda(lambda x: K.l2_normalize(x, axis=-1))(new_output)

    model = keras.engine.training.Model(model.inputs, new_output)
    return model

def generator(x, y, k, BATCH_SIZE):

    n_img = 0
    while True:
        x_in = []
        y_in= []
        for n in range(BATCH_SIZE*k):

            if n_img >= len(x):
                x, y = shuffle(x, y)
                n_img = 0

            image = io.imread(x[n_img])
            image_augmented = data_augmentation(image)

            x_in.append(image_augmented)
            y_in.append(y[n_img])

            n_img  += 1

        x_in, y_in = np.asarray(x_in), np.asarray(y_in)
        yield x_in, y_in

