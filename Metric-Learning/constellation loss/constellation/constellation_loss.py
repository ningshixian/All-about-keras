import tensorflow as tf

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """

    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def constellation(k, BATCH_SIZE):

    def c_loss(labels, embeddings):
        """Build the constellation loss over a batch of embeddings.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            ctl_loss: scalar tensor containing the constellation loss
        """

        labels = labels[:, 0]

        labels_list = []
        embeddings_list = []
        for i in range(k):
            labels_list.append(labels[BATCH_SIZE * i:BATCH_SIZE * (i + 1)])
            embeddings_list.append(embeddings[BATCH_SIZE * i:BATCH_SIZE * (i + 1)])

        loss_list = []
        for i in range(len(embeddings_list)):
            # Get the dot product
            pairwise_dist = tf.matmul(embeddings_list[i], tf.transpose(embeddings_list[i]))

            # shape (batch_size, batch_size, 1)
            anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
            assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
            # shape (batch_size, 1, batch_size)
            anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
            assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

            ctl_loss = anchor_negative_dist - anchor_positive_dist

            # (where label(a) != label(p) or label(n) == label(a) or a == p)
            mask = _get_triplet_mask(labels_list[i])
            mask = tf.to_float(mask)
            ctl_loss = tf.multiply(mask, ctl_loss)

            loss_list.append(ctl_loss)

        ctl_loss = 1. + tf.exp(loss_list[0])
        for i in range(1, len(embeddings_list)):
            ctl_loss += tf.exp(loss_list[i])

        ctl_loss = tf.log(ctl_loss)

        # # Get final mean constellation loss and divide due to very large loss value
        ctl_loss = tf.reduce_sum(ctl_loss) / 1000.

        return ctl_loss

    return c_loss

