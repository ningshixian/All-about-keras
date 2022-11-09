import keras
import matplotlib as plt
import glob
import numpy as np
from sklearn.utils import shuffle
import skimage
import random
import skimage.io as io

def load_images(x):

    x_input = []
    for i in range(len(x)):
        image = io.imread(x[i])
        image_augmented = data_augmentation(image)
        x_input.append(image_augmented)

    return np.asarray(x_input)


def load_data(dataset_path):
    '''
        You can obtain the dataset here: https://www.nature.com/articles/srep27988
    '''

    # Nature images here
    folders = glob.glob(dataset_path + '/*')
    N_CAT_TOT = len(folders)

    x = []
    y = []
    n_cat = 0
    for folder in folders:
        image_paths = glob.glob(folder + '/*.tif')
        for img in image_paths:
            x.append(img)
            y.append(n_cat)
        n_cat += 1

    x, y = np.asarray(x), np.asarray(y)
    x, y = shuffle(x, y, random_state=666)

    return x, y


def data_augmentation(img):

    a =  random.randint(0,1)
    b = random.randint(0,3)

    image_a = [img, img[::-1,:,:]]
    angle_a = [0,90,180,270]

    image_transformed = skimage.transform.rotate(image_a[a],angle=angle_a[b])
    return preprocess_input(image_transformed)

def preprocess_input(x):
    x -= 0.5
    x *= 2.
    return x
