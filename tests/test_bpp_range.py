import unittest
import tensorflow as tf
import numpy as np
from models.bpp_range import LogarithmicOrLinearFit


class BppRangeTestCase(unittest.TestCase):
    def test_logarithmic_or_linear_fit(self) -> None:
        sess = tf.keras.backend.get_session()

        alpha_to_bpp = LogarithmicOrLinearFit()
        sess.run(tf.global_variables_initializer())

        alpha_to_bpp.fit((0.01, 0.1), (0.75, 2.0))
        intercept = (0.1 - 0.01) / (2.0 - 0.75)
        np.testing.assert_almost_equal(alpha_to_bpp.inverse_numpy(np.array([0.75, 1., 1.5, 2.])),
                                       np.array([0.01, 0.01 + intercept * 0.25, 0.01 + intercept * 0.75, 0.1]),
                                       err_msg='The initial fit was not linear')

        bpps = np.linspace(0.5, 3, 10000).astype(np.float32)
        alphas = 5 * np.exp(10 * bpps - 0.1) + 0.35
        alphas = alphas.astype(np.float32)

        alpha_to_bpp.fit(alphas, np.repeat(2.2, 10000))
        self.assertLess(alpha_to_bpp.inverse_numpy(np.array([0.5]))[0] + 0.1,
                        alpha_to_bpp.inverse_numpy(np.array([1.0]))[0],
                        'Did not spread out on equal bpp for different alphas')
        np.testing.assert_allclose(alpha_to_bpp.inverse_numpy(np.array([0.5, 1.0, 1.5])),
                                   sess.run(alpha_to_bpp.inverse_tf(tf.constant([0.5, 1.0, 1.5]))),
                                   err_msg='Tf and numpy version are different for equal bpp',
                                   rtol=1e-5)

        alpha_to_bpp.fit(alphas, bpps)
        np.testing.assert_allclose(bpps, alpha_to_bpp.forward_numpy(alphas),
                                   err_msg='Did not fit a forward exponential function correctly',
                                   rtol=1e-5)
        np.testing.assert_allclose(alphas, alpha_to_bpp.inverse_numpy(bpps),
                                   err_msg='Did not fit an inverse exponential function correctly',
                                   rtol=1e-5)
        np.testing.assert_allclose(alpha_to_bpp.forward_numpy(alphas),
                                   sess.run(alpha_to_bpp.forward_tf(tf.constant(alphas))),
                                   err_msg='Forward function on exponential data is different between Numpy and TF',
                                   rtol=1e-5)
        np.testing.assert_allclose(alpha_to_bpp.inverse_numpy(bpps),
                                   sess.run(alpha_to_bpp.inverse_tf(tf.constant(bpps))),
                                   err_msg='Inverse function on exponential data is different between Numpy and TF',
                                   rtol=1e-5)

        alphas = 1.77 * bpps - 0.12
        alpha_to_bpp.fit(alphas, bpps)
        np.testing.assert_allclose(bpps, alpha_to_bpp.forward_numpy(alphas),
                                   err_msg='Did not fit a forward linear function correctly',
                                   rtol=1e-5)
        np.testing.assert_allclose(alphas, alpha_to_bpp.inverse_numpy(bpps),
                                   err_msg='Did not fit an inverse linear function correctly',
                                   rtol=1e-5)
        np.testing.assert_allclose(alpha_to_bpp.forward_numpy(alphas),
                                   sess.run(alpha_to_bpp.forward_tf(tf.constant(alphas))),
                                   err_msg='Forward function on linear data is different between Numpy and TF',
                                   rtol=1e-5)
        np.testing.assert_allclose(alpha_to_bpp.inverse_numpy(bpps),
                                   sess.run(alpha_to_bpp.inverse_tf(tf.constant(bpps))),
                                   err_msg='Inverse function on linear data is different between Numpy and TF',
                                   rtol=1e-5)
