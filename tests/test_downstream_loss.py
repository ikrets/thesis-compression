import tensorflow as tf
import numpy as np
import unittest
from models.downstream_losses import PerceptualLoss

tfk = tf.keras
tfkl = tf.keras.layers


class DownstreamTaskTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_metric = lambda label, x: tf.constant(0.)

    def test_deep_perceptual_loss(self):
        sess = tf.keras.backend.get_session()

        model = tfk.Sequential([
            tfkl.Input((6, 6, 3)),
            tfkl.Lambda(lambda x: x * 2, name='readout_1'),
            tfkl.Lambda(lambda x: tf.tile(x, multiples=(1, 1, 1, 2))),
            tfkl.Lambda(lambda x: x),
            tfkl.Lambda(lambda x: x[:, 1:-1, 1:-1, :] * 3, name='readout_2'),
            tfkl.Lambda(lambda x: tf.tile(x, multiples=(1, 1, 1, 2))),
            tfkl.Lambda(lambda x: x[:, 1:-1, 1:-1, :] - 2, name='readout_3'),
            tfkl.AveragePooling2D(),
            tfkl.Flatten(),
            tfkl.Dense(10, activation='softmax', name='final_dense')
        ])

        readouts = ['readout_1', 'readout_2', 'readout_3']
        loss = PerceptualLoss(readout_layers=readouts, model=model,
                              metric_fn=self.dummy_metric,
                              preprocess_fn=lambda x: x,
                              normalize_activations=False)

        X = tf.ones((3, 6, 6, 3))
        X_rec = tf.ones((3, 6, 6, 3)) / 2.

        loss_value = loss.loss(X, X_rec)
        np.testing.assert_almost_equal(loss_value.shape.as_list(), (3,))

        # computed by hand
        readout_1_loss = [1] * 3
        readout_2_loss = [9] * 6
        readout_3_loss = [9] * 12
        total_loss = np.mean(np.concatenate([readout_1_loss, readout_2_loss, readout_3_loss]))

        np.testing.assert_almost_equal([total_loss] * 3,
                                       sess.run(loss_value))

        loss = PerceptualLoss(readout_layers=['readout_1'],
                              model=model,
                              metric_fn=self.dummy_metric,
                              normalize_activations=False,
                              preprocess_fn=lambda x: x)
        loss_value = loss.loss(X, X_rec)
        np.testing.assert_almost_equal(loss_value.shape.as_list(), (3, ))
        np.testing.assert_almost_equal(np.mean([readout_1_loss] * 3, axis=1), sess.run(loss_value))


if __name__ == '__main__':
    unittest.main()
