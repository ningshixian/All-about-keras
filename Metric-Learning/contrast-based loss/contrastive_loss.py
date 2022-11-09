# import the necessary packages
import keras.backend as K
import tensorflow as tf

def contrastive_loss(y_true, y_pred, margin = 1):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # calculate the contrastive loss between the true labels and
	# the predicted labels
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

model.compile(loss=contrastive_loss, ...)