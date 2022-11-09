
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

from multiclass_loss import npairs_loss
from utils import load_data, load_images
from network import inception, generator


if __name__ == '__main__':

    # PARAMETERS
    IMG_SIZE = 150
    EMB_VECTOR = 128

    DATASET_PATH = '/mnt/RAID5/users/alfonsomedela/projects/piccolo/nature/NATURE'

    # LOAD THE DATA
    x, y = load_data(DATASET_PATH)

    random_seeds = [666, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    accuracy = []
    silhouette = []
    davis = []
    bac_list = []

    fold = 0
    for seed in random_seeds:

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

        x_train_input = load_images(x_train)
        x_test_input = load_images(x_test)

        print('Train data:', x_train.shape, y_train.shape)
        print('Test data:', x_test.shape, y_test.shape)

        print('')
        print('Training...')

        # TRAIN
        gpu_device = "/gpu:0"  # 0,1,2,3
        if keras.backend.backend() == 'tensorflow':
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
            session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_config.gpu_options.allow_growth = True
            session = K.tf.Session(config=session_config)
            with K.tf.device(gpu_device):
                # DEFINE THE MODEL
                model = inception(EMB_VECTOR, IMG_SIZE)

                model.compile(loss=npairs_loss, optimizer=keras.optimizers.Adam(1e-3))
                model.load_weights('final_exp/weights/inception_multiclass_fold_' + str(fold) + '.h5')

                embeddings_train = model.predict([x_train_input])
                embeddings_test = model.predict([x_test_input])

                'ML model to train with extracted feature vectors'

                KNN = KNeighborsClassifier()
                KNN.fit(embeddings_train, y_train)
                score = KNN.score(embeddings_test, y_test)

                accuracy.append(score * 100.)

                y_pred = KNN.predict(embeddings_test)
                BAC = balanced_accuracy_score(y_test, y_pred) * 100.
                bac_list.append(BAC)

                'Homogeneity test'
                d_score = davies_bouldin_score(embeddings_test, y_test)
                s_score = silhouette_score(embeddings_test, y_test)

                silhouette.append(s_score)
                davis.append(d_score)

                session.close()
                fold += 1

    accuracy = np.asarray(accuracy)
    davis = np.asarray(davis)
    silhouette = np.asarray(silhouette)
    bac_list = np.asarray(bac_list)

    # Get results from list of crossvalidations
    mean_accuracy, std_accuracy = np.mean(accuracy), np.std(accuracy)
    mean_s, std_s = np.mean(silhouette), np.std(silhouette)
    mean_d, std_d = np.mean(davis), np.std(davis)
    mean_bac, std_bac = np.mean(bac_list), np.std(bac_list)

    mean_accuracy, std_accuracy = np.reshape(mean_accuracy, (1)), np.reshape(std_accuracy, (1))
    mean_s, std_s = np.reshape(mean_s, (1)), np.reshape(std_s, (1))
    mean_d, std_d = np.reshape(mean_d, (1)), np.reshape(std_d, (1))
    mean_bac, std_bac = np.reshape(mean_bac, (1)), np.reshape(std_bac, (1))

    # Save the data
    accuracy_row = np.concatenate((accuracy, mean_accuracy, std_accuracy), axis=0)
    accuracy_row = np.reshape(accuracy_row, (len(accuracy_row), 1))
    accuracy_row = np.around(accuracy_row, decimals=2)

    silhouette_row = np.concatenate((silhouette, mean_s, std_s), axis=0)
    silhouette_row = np.reshape(silhouette_row, (len(silhouette_row), 1))
    silhouette_row = np.around(silhouette_row, decimals=4)

    davis_row = np.concatenate((davis, mean_d, std_d), axis=0)
    davis_row = np.reshape(davis_row, (len(davis_row), 1))
    davis_row = np.around(davis_row, decimals=4)

    bac_row = np.concatenate((bac_list, mean_bac, std_bac), axis=0)
    bac_row = np.reshape(bac_row, (len(bac_row), 1))
    bac_row = np.around(bac_row, decimals=2)

    csv_array = np.concatenate((accuracy_row, bac_row, davis_row, silhouette_row), axis=-1)

    df = pd.DataFrame(csv_array, columns=['accuracy', 'BAC', 'Davis', 'Silhouette'])

    df.to_csv('final_exp/results/final_exp_8_pair_mc.csv')












































