
# Alfonso Medela & Artzai Picon, "Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding.", submitted to NeurIPS 2019.

import os
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import davies_bouldin_score, silhouette_score, balanced_accuracy_score
from sklearn.manifold import TSNE

from constellation_loss import constellation
from utils import load_data, load_images
from network import inception, generator


if __name__ == '__main__':

    DATASET_PATH = '/mnt/RAID5/users/alfonsomedela/projects/piccolo/nature/NATURE'

    # PARAMETERS
    IMG_SIZE = 150
    EMB_VECTOR = 128
    BATCH_SIZE = 32
    k = 6

    # LOAD THE DATA
    x, y = load_data(DATASET_PATH)

    fold = 5  # Choose fold
    random_seeds = [666, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    SEED = random_seeds[fold]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    x_test_input = load_images(x_test)

    # TRAIN
    gpu_device = "/gpu:2"  # 0,1,2,3
    if keras.backend.backend() == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = K.tf.Session(config=session_config)
        with K.tf.device(gpu_device):

            # DEFINE THE MODEL
            model = inception(EMB_VECTOR, IMG_SIZE)

            model.compile(loss=constellation(k, BATCH_SIZE), optimizer=keras.optimizers.Adam(1e-3))
            model.load_weights('final_exp/' + str(k) + '/weights/constellation_fold_' + str(fold) + '.h5')

            embeddings_test = model.predict([x_test_input])

            x_tsne = TSNE(n_components=2).fit_transform(embeddings_test)

            color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', (1., 170. / 255., 58. / 255.)]
            labels = ['Empty', 'Lympho', 'Mucosa', 'Stroma', 'Tumor', 'Complex', 'Debris', 'Adipose']
            ind = [0 for i in range(len(np.unique(y_test)))]

            for i in range(len(embeddings_test)):
                if ind[y_test[i]] == 0:
                    plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y_test[i]], label=labels[y_test[i]])
                    ind[y_test[i]] = 1
                else:
                    plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y_test[i]])

            plt.ylabel(r'$z_2$')
            plt.xlabel(r'$z_1$')
            # plt.legend()
            plt.savefig('tsne.png')

            session.close()










































