
# Alfonso Medela & Artzai Picon, "Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding.", submitted to NeurIPS 2019.

import os
import keras
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from multiclass_loss import npairs_loss
from utils import load_data
from network import inception, generator


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.savefig('final_exp/plots/inception_multiclass_fold_' + str(fold) + '.png');


if __name__ == '__main__':

    loss_plot = PlotLosses()

    DATASET_PATH = '/mnt/RAID5/users/alfonsomedela/projects/piccolo/nature/NATURE'

    # PARAMETERS
    BATCH_SIZE = 16
    IMG_SIZE = 150
    EMB_VECTOR = 128
    EPOCHS = 10

    # LOAD THE DATA
    x, y = load_data(DATASET_PATH)

    random_seeds = [666, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    fold = 0
    for seed in random_seeds:

        # 80% - 20%
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

        steps_train = int(len(x_train) * 1. / BATCH_SIZE ) + 1
        steps_test = int(len(x_test) * 1. / BATCH_SIZE ) + 1

        print('Fold: ' + str(fold))

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

                for layer in model.layers:
                    if isinstance(layer, keras.layers.BatchNormalization):
                        layer.momentum = 0.9

                for layer in model.layers[:-50]:
                    if not isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = False

                # TRAIN THE MODEL

                model.compile(loss=npairs_loss, optimizer=keras.optimizers.Adam(1e-3))

                model.fit_generator(generator(x_train, y_train), steps_per_epoch=steps_train,
                                    validation_data=generator(x_test, y_test), validation_steps=steps_test,
                                    epochs=EPOCHS, callbacks=[loss_plot])

                model.save_weights('final_exp/weights/inception_multiclass_fold_' + str(fold) + '.h5')

                # close session and add a fold
                session.close()
                fold += 1








































