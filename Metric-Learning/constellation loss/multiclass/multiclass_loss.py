from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.summary import summary

'Inplementation in tensorflow by: https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py'

def npairs_loss(labels, embeddings, BATCH_SIZE = 16, reg_lambda=0.002, print_losses=False):

    labels = labels[:BATCH_SIZE//2, 0]
    embeddings_anchor = embeddings[:BATCH_SIZE // 2, :]
    embeddings_positive = embeddings[BATCH_SIZE // 2:, :]

    # pylint: enable=line-too-long
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(
        0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(
        embeddings_anchor, embeddings_positive, transpose_a=False,
        transpose_b=True)

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    labels_remapped = math_ops.to_float(
        math_ops.equal(labels, array_ops.transpose(labels)))
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keepdims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(
        logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(
            xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss
