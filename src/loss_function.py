import tensorflow as tf


def micropsi_loss_batch(y_true, y_pred):
    """Custom loss function

    The function aims to minimize euclidean distances of each predicted vector to a target vector.
    If M target vectors with corresponding estimates is passed to the function, M loss values is returned.

    Parameters
    ----------
    y_true : tf.Tensor
        Target 3D vector(s)
    y_pred : tf.Tensor
        Model returns predictions as K estimates for each target vector.

    Returns
    -------
    tf.Tensor
        If inputs are single vector and its estimates, a tensor with loss is returned
    List<tf.Tensor>
        Every tensor value in a list corresponds to a loss of each vector (and its estimates)

    Raises
    ------
    ValueError
        If a inputs' batch sizes don't match or the last dimension is not 3.
    TypeError
        If the input tensors aren't floats.
    """
    if not y_pred.dtype.is_floating:
        raise TypeError("tensor y_pred type must be float instead %s" % y_pred.dtype)

    if not y_true.dtype.is_floating:
        raise TypeError("tensor y_true type must be float insted %s" % y_true.dtype)

    if y_true.shape.dims[y_true.shape.ndims - 1] != 3:
        raise ValueError("last dimension of y_true has to be 3")

    if y_pred.shape.dims[y_pred.shape.ndims - 1] != 3:
        raise ValueError("last dimension of y_pred has to be 3")

    if y_pred.shape.ndims == 2 and y_true.shape.ndims == 1:
        # single vector and it's estimates
        return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(y_true, y_pred), 2), axis=1)))

    if y_pred.shape.dims[0] != y_true.shape.dims[0]:
        raise ValueError("batch size of y_true should equal batch size y_pred but got %d, %d" % (y_true.shape.dims[0],
                                                                                                 y_pred.shape.dims[0]))

    #: int: Batch size
    M = y_pred.shape.dims[0]

    y_pred_sliced = tf.split(y_pred, num_or_size_splits=M, axis=0)
    y_true_sliced = tf.split(y_true, num_or_size_splits=M, axis=0)
    loss_batch = []

    for i in range(M):
        loss_batch.append(tf.reduce_sum(tf.sqrt(tf.reduce_sum(
            tf.pow(tf.subtract(y_true_sliced[i][0], y_pred_sliced[i][0]), 2), axis=1))))

    return loss_batch
