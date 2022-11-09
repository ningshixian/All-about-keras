import keras
from keras.layers import Lambda
import keras.backend as K
from sklearn.utils import shuffle
import skimage.io as io
import numpy as np

from utils import data_augmentation

def inception(EMB_VECTOR, IMG_SIZE, use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)

    # # add new dense layer for our labels
    new_output = keras.layers.Dense(EMB_VECTOR, activation='sigmoid')(new_output)

    model = keras.engine.training.Model(model.inputs, new_output)
    return model

def generator(x, y):

    x_class = [[] for i in range(len(np.unique(y)))]
    for i in range(len(x)):
        x_class[y[i]].append(x[i])

    ind = [0 for i in range(len(np.unique(y)))]
    y_class = [i for i in range(len(np.unique(y)))]

    while True:
        x_in = []
        y_in= []

        x_class, ind, y_class = shuffle(x_class, ind, y_class)

        for n in range(2):

            for i in range(len(ind)):
                if ind[i] >= len(x_class[i]):
                    x_class[i] = shuffle(x_class[i])
                    ind[i] = 0

            for i in range(len(ind)):
                image = io.imread(x_class[i][ind[i]])
                image_augmented = data_augmentation(image)

                x_in.append(image_augmented)
                y_in.append(y_class[i])

                ind[i] += 1

        x_in, y_in = np.asarray(x_in), np.asarray(y_in)

        yield x_in, y_in
