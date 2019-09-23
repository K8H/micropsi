import unittest

import tensorflow as tf

import src.loss_function as loss_function


class LossFunctionTests(unittest.TestCase):
    Y_TRUE = [1, 5, 8]
    Y_PRED = [[2, 3.8, 6.7], [0, 1.5, 7.7], [-1, 5, 8]]
    Y_TRUE_BATCH = [[1, 7, 9], [5, -2, -4], [1, 0, 1], [-1, 6, 3]]
    Y_TRUE_BATCH_SMALL = [[1, 7, 9], [5, -2, -4], [1, 0, 1]]
    Y_TRUE_2D = [[1, 7], [5, -2], [1, 0]]
    Y_PRED_BATCH = [[[1.5, 6, 10.2], [0.3, 6.6, -1], [1, 7, 9], [1.5, 6, 10.2], [0.3, 6.6, -1], [1, 7, 9]],
                    [[5.6, -2.1, -3], [6.2, 2.5, 4.5], [-1, -2, -4], [5.6, -2.1, -3], [6.2, 2.5, 4.5], [-1, -2, -4]],
                    [[1, 0, 1], [1.5, -0.3, 0.7], [1.1, 0.3, -1], [1, 0, 1], [1.5, -0.3, 0.7], [1.1, 0.3, -1]],
                    [[-1, 6, 3], [-1.3, 5.6, 3.2], [1, 6.2, 3.1], [-1, 6, 3], [-1.3, 5.6, 3.2], [1, 6.2, 3.1]]]

    Y_TRUE_TF = tf.constant(Y_TRUE, tf.float32)
    Y_PRED_TF = tf.constant(Y_PRED, tf.float32)
    Y_TRUE_BATCH_TF = tf.constant(Y_TRUE_BATCH, tf.float32)
    Y_TRUE_TYPE_TF = tf.constant(Y_TRUE_BATCH, tf.int16)
    Y_TRUE_SMALL_TF = tf.constant(Y_TRUE_BATCH_SMALL, tf.float32)
    Y_TRUE_2D_TF = tf.constant(Y_TRUE_2D, tf.float32)
    Y_PRED_BATCH_TF = tf.constant(Y_PRED_BATCH, tf.float32)

    def test_micropsi_loss_single(self):
        loss = loss_function.micropsi_loss_batch(y_true=self.Y_TRUE_TF, y_pred=self.Y_PRED_TF)

        with tf.Session() as sess:
            loss = sess.run(loss)

        self.assertEqual(7.684637, float("%.6f" % loss))

    def test_micropsi_loss_batch(self):
        loss = loss_function.micropsi_loss_batch(y_true=self.Y_TRUE_BATCH_TF, y_pred=self.Y_PRED_BATCH_TF)

        with tf.Session() as sess:
            loss = sess.run(loss)

        self.assertEqual(23.34514, float("%.5f" % loss[0]))

    def test_micropsi_loss_value_error(self):
        self.assertRaises(ValueError, loss_function.micropsi_loss_batch, self.Y_TRUE_SMALL_TF, self.Y_PRED_BATCH_TF)

    def test_micropsi_loss_type_error(self):
        self.assertRaises(TypeError, loss_function.micropsi_loss_batch, self.Y_TRUE_TYPE_TF, self.Y_PRED_BATCH_TF)

    def test_micropsi_loss_dim_error(self):
        self.assertRaises(ValueError, loss_function.micropsi_loss_batch, self.Y_TRUE_2D_TF, self.Y_PRED_BATCH_TF)


if __name__ == "__main__":
    unittest.main()
