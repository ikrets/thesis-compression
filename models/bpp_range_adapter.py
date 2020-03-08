import tensorflow as tf
import numpy as np
from models.compressors import SimpleFiLMCompressor, bits_per_pixel
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
from typing import Tuple, Sequence, Dict


class BppToAlphaFn:
    def __init__(self,
                 initial_alpha_range: Tuple[float, float],
                 bpp_range: Tuple[float, float]) -> None:
        initial_a = (bpp_range[0] - bpp_range[1]) / (np.log(initial_alpha_range[0]) - np.log(initial_alpha_range[1]))
        initial_b = 1
        initial_c = 0
        initial_d = bpp_range[0] - initial_a * np.log(initial_alpha_range[0])

        self.opts = tf.Variable([initial_a, initial_b, initial_c, initial_d],
                                name='bpp_to_alpha_opts',
                                trainable=False,
                                dtype=tf.float32,
                                shape=(4,))
        self.opt_placeholders = tf.placeholder(tf.float32, shape=(4,))
        self.assign_opts = tf.assign(self.opts, self.opt_placeholders)

        self.sess = tf.keras.backend.get_session()
        self.sess.run(tf.variables_initializer([self.opts]))

    def fit(self, bpps: Sequence[float], alphas: Sequence[float]) -> None:
        def f(x, a, b, c, d):
            return a * np.log(b * x + c) + d

        popt, _ = curve_fit(f, alphas, bpps)
        self.sess.run(self.assign_opts, {self.opt_placeholders: popt})

    def __call__(self, bpp: tf.Tensor) -> tf.Tensor:
        value = (tf.exp(bpp / self.opts[0]) / tf.exp(self.opts[3] / self.opts[0]) - self.opts[2]) / self.opts[1]
        return tf.maximum(0., value)

    def numpy_call(self, bpp: np.array) -> np.array:
        a, b, c, d = self.sess.run(self.opts)
        return np.maximum(0., (np.exp(bpp / a) / np.exp(d / a) - c) / b)


class BppRangeAdapter:
    def __init__(self, compressor: SimpleFiLMCompressor,
                 eval_dataset: tf.data.Dataset,
                 eval_dataset_steps: int,
                 bpp_range: Tuple[float, float],
                 lmbda: float,
                 initial_alpha_range: Tuple[float, float],
                 alpha_linspace_steps: int
                 ) -> None:
        self.bpp_range = bpp_range
        self.compressor = compressor
        self.eval_dataset = eval_dataset
        self.eval_dataset_steps = eval_dataset_steps
        self.lmbda = lmbda
        self.alpha_linspace_steps = alpha_linspace_steps
        self.bpp_linspace = np.linspace(*self.bpp_range, self.alpha_linspace_steps)

        self._alpha = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._alpha_placeholder = tf.placeholder(tf.float32)
        self._assign_alpha = tf.assign(self._alpha, self._alpha_placeholder)

        self.bpp_to_alpha = BppToAlphaFn(bpp_range=bpp_range, initial_alpha_range=initial_alpha_range)

        batch = tf.compat.v1.data.make_one_shot_iterator(self.eval_dataset).get_next()
        batch = self._add_parameters(batch)
        compressor_output = self.compressor.forward(batch, training=False)
        self.bpp = bits_per_pixel(compressor_output['Y_likelihoods'], batch['X'].shape)

        sess = tf.keras.backend.get_session()
        sess.run(tf.variables_initializer([self._alpha]))

    def _update_current_alpha_range(self) -> None:
        alpha_linspace = self.bpp_to_alpha.numpy_call(self.bpp_linspace)
        self.current_alpha_range = alpha_linspace[0], alpha_linspace[-1]

    def _add_parameters(self, dataset_item: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(dataset_item['X'])[0]
        dataset_item.update({
            'lambda': tf.repeat(self.lmbda, batch_size),
            'alpha': tf.repeat(self._alpha, batch_size)
        })

        return dataset_item

    def _compute_alpha_bpp(self) -> Tuple[np.array, np.array]:
        sess = tf.keras.backend.get_session()
        alpha_linspace = self.bpp_to_alpha.numpy_call(self.bpp_linspace)

        alphas = []
        mean_bpps = []
        for alpha in tqdm(alpha_linspace, desc='Evaluating alphas'):
            sess.run(self._assign_alpha, {self._alpha_placeholder: alpha})
            bpps = []
            for _ in trange(self.eval_dataset_steps):
                bpps.extend(sess.run(self.bpp))

            alphas.append(alpha)
            mean_bpps.append(np.mean(bpps))

        return alphas, mean_bpps

    def update(self) -> None:
        alphas, mean_bpps = self._compute_alpha_bpp()
        self.bpp_to_alpha.fit(mean_bpps, alphas)
        self._update_current_alpha_range()

    def add_computed_parameters(self, dataset_item: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(dataset_item['X'])[0]
        random_bpps = tf.random.uniform((batch_size,), self.bpp_range[0], self.bpp_range[1])
        random_alphas = self.bpp_to_alpha(random_bpps)

        dataset_item.update({
            'lambda': tf.repeat(self.lmbda, batch_size),
            'alpha': tf.maximum(0., random_alphas)
        })

        return dataset_item
