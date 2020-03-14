import json

import tensorflow as tf
import pandas as pd
import numpy as np
from models.compressors import SimpleFiLMCompressor, bits_per_pixel, pipeline_add_constant_parameters
from scipy.optimize import curve_fit
from sklearn.metrics import auc
from tqdm import tqdm, trange
from typing import Tuple, Sequence, Dict, List, TextIO

from models.downstream_losses import PerceptualLoss

_log_eps = 1e-5


def _non_negative_log_curve(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a * np.log(np.maximum(b * x + c, _log_eps)) + d


def _non_negative_linear_curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


class LogarithmicOrLinearFit:
    def __init__(self, linear_version=True, linear_opts=(0., 0.), log_opts=(0., 0., 0., 0.)) -> None:
        self.log_opts = log_opts
        self.log_opts_var = tf.Variable(self.log_opts,
                                        name='log_opts',
                                        trainable=False,
                                        dtype=tf.float32,
                                        shape=(4,))
        self.log_opt_placeholders = tf.placeholder(tf.float32, shape=(4,))
        self.assign_log_opts = tf.assign(self.log_opts_var, self.log_opt_placeholders)

        self.linear_opts = linear_opts
        self.linear_opts_var = tf.Variable(self.linear_opts, name='linear_opts', trainable=False, dtype=tf.float32,
                                           shape=(2,))
        self.linear_opt_placeholders = tf.placeholder(tf.float32, shape=(2,))
        self.assign_linear_opts = tf.assign(self.linear_opts_var, self.linear_opt_placeholders)

        self.linear_version = linear_version
        self.linear_version_var = tf.Variable(True, name='linear_version', dtype=tf.bool, trainable=False,
                                              shape=())
        self.linear_version_placeholder = tf.placeholder(tf.bool, shape=())
        self.assign_linear_version = tf.assign(self.linear_version_var, self.linear_version_placeholder)

        self.sess = tf.keras.backend.get_session()

    def fit(self, x: Sequence[float], y: Sequence[float]) -> None:
        if len(x) != len(y):
            raise ValueError('Length of x and y should be same.')

        try:
            if len(x) < 4:
                raise RuntimeError

            popt, _ = curve_fit(_non_negative_log_curve, x, y)
            self.sess.run(self.assign_log_opts, {self.log_opt_placeholders: popt})
            self.sess.run(self.assign_linear_version, {self.linear_version_placeholder: False})
            self.linear_version = False
            self.log_opts = popt
        except RuntimeError:
            try:
                if len(x) < 2:
                    raise RuntimeError
                popt, _ = curve_fit(_non_negative_linear_curve, x, y)
            except RuntimeError:
                popt = [1., 0.]

            if abs(popt[0]) < 1e-6:
                # virtually no change in Y depending on X, use the fallback a=1 b=0
                popt = [1., 0.]

            self.sess.run(self.assign_linear_opts, {self.linear_opt_placeholders: popt})
            self.sess.run(self.assign_linear_version, {self.linear_version_placeholder: True})
            self.linear_version = True
            self.linear_opts = popt

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        if self.linear_version:
            return _non_negative_linear_curve(x, *self.linear_opts)
        else:
            return _non_negative_log_curve(x, *self.log_opts)

    def forward_tf(self, x: tf.Tensor) -> tf.Tensor:
        def linear_fn():
            a = self.linear_opts_var[0]
            b = self.linear_opts_var[1]
            return tf.maximum(0., a * x + b)

        def log_fn():
            a = self.log_opts_var[0]
            b = self.log_opts_var[1]
            c = self.log_opts_var[2]
            d = self.log_opts_var[3]
            return tf.maximum(0., a * tf.log(tf.maximum(_log_eps, b * x + c)) + d)

        return tf.cond(self.linear_version_var, true_fn=linear_fn, false_fn=log_fn)

    def inverse_numpy(self, y: np.ndarray) -> np.ndarray:
        if self.linear_version:
            a, b = self.linear_opts
            return (np.maximum(y, 0.) - b) / a
        else:
            a, b, c, d = self.log_opts
            return np.maximum((np.exp((y - d) / a) - c) / b, 0.)

    def inverse_tf(self, y: tf.Tensor) -> tf.Tensor:
        def linear_fn():
            a = self.linear_opts_var[0]
            b = self.linear_opts_var[1]
            return tf.maximum((y - b) / a, 0.)

        def log_fn():
            a = self.log_opts_var[0]
            b = self.log_opts_var[1]
            c = self.log_opts_var[2]
            d = self.log_opts_var[3]

            return tf.maximum((tf.exp((y - d) / a) - c) / b, 0.)

        return tf.cond(self.linear_version_var, true_fn=linear_fn, false_fn=log_fn)

    def save(self, fp):
        json.dump({'linear_version': self.linear_version,
                   'opts': tuple(self.linear_opts if self.linear_version else self.log_opts)}, fp)

    @staticmethod
    def load(fp: TextIO) -> 'LogarithmicOrLinearFit':
        params = json.load(fp)

        if params['linear_version']:
            return LogarithmicOrLinearFit(linear_version=True, linear_opts=params['opts'])
        else:
            return LogarithmicOrLinearFit(linear_version=False, log_opts=params['opts'])


class BppRangeAdapter:
    def __init__(self, compressor: SimpleFiLMCompressor,
                 eval_dataset: tf.data.Dataset,
                 eval_dataset_steps: int,
                 bpp_range: Tuple[float, float],
                 lmbda: float,
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

        self.alpha_to_bpp = LogarithmicOrLinearFit()
        self._update_current_alpha_linspace()

        batch = tf.compat.v1.data.make_one_shot_iterator(self.eval_dataset).get_next()
        batch = pipeline_add_constant_parameters(batch, self._alpha, self.lmbda)
        compressor_output = self.compressor.forward(batch, training=False)
        self.bpp = bits_per_pixel(compressor_output['Y_likelihoods'], batch['X'].shape)

    def _update_current_alpha_linspace(self) -> None:
        self.current_alpha_linspace = self.alpha_to_bpp.inverse_numpy(self.bpp_linspace)

    def _compute_alpha_bpp(self) -> Tuple[np.array, np.array]:
        sess = tf.keras.backend.get_session()
        alpha_linspace = self.alpha_to_bpp.inverse_numpy(self.bpp_linspace)

        alphas = []
        mean_bpps = []
        for alpha in tqdm(alpha_linspace, desc='Evaluating alphas'):
            sess.run(self._assign_alpha, {self._alpha_placeholder: alpha})
            bpps: List[np.array] = []
            for _ in trange(self.eval_dataset_steps):
                bpps.extend(sess.run(self.bpp))

            alphas.append(alpha)
            mean_bpps.append(np.mean(bpps))

        return alphas, mean_bpps

    def update(self) -> None:
        alphas, mean_bpps = self._compute_alpha_bpp()
        self.alpha_to_bpp.fit(alphas, mean_bpps)
        self._update_current_alpha_linspace()

    def add_computed_parameters(self, dataset_item: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(dataset_item['X'])[0]
        random_bpps = tf.random.uniform((batch_size,), self.bpp_range[0], self.bpp_range[1])
        random_alphas = self.alpha_to_bpp.inverse_tf(random_bpps)

        dataset_item.update({
            'lambda': tf.repeat(self.lmbda, batch_size),
            'alpha': tf.maximum(0., random_alphas)
        })

        return dataset_item


class ImageAlphaComparison:
    alphas: List[float] = []
    images: List[np.array] = []
    original_image: np.array


class BppRangeEvaluation:
    def __init__(self,
                 compressor: SimpleFiLMCompressor,
                 downstream_loss: PerceptualLoss,
                 bpp_range_adapter: BppRangeAdapter,
                 val_dataset: tf.data.Dataset,
                 val_dataset_steps: int,
                 ) -> None:
        self.compressor = compressor
        self.downstream_loss = downstream_loss
        self.bpp_range_adapter = bpp_range_adapter
        self.val_dataset = val_dataset
        self.val_dataset_steps = val_dataset_steps

        self.alpha_placeholder = tf.placeholder(tf.float32)
        batch = self.val_dataset.make_one_shot_iterator().get_next()
        batch = pipeline_add_constant_parameters(batch,
                                                 alpha=self.alpha_placeholder,
                                                 lmbda=self.bpp_range_adapter.lmbda)
        compressor_outputs = self.compressor.forward(batch, training=False)
        clipped_X_tilde = tf.clip_by_value(compressor_outputs['X_tilde'], 0, 1)

        self.eval_results = {'bpp': bits_per_pixel(compressor_outputs['Y_likelihoods'], tf.shape(batch['X'])),
                             'psnr': tf.image.psnr(batch['X'], clipped_X_tilde, max_val=1.),
                             'downstream_loss': self.downstream_loss.loss(X=batch['X'],
                                                                          X_reconstruction=clipped_X_tilde),
                             'downstream_metric': self.downstream_loss.metric(label=batch['label'],
                                                                              X_reconstruction=clipped_X_tilde),
                             'X': batch['X'],
                             'alpha': batch['alpha'],
                             'lambda': batch['lambda'],
                             'X_tilde': clipped_X_tilde}

    def evaluate(self, num_alpha_comparisons_pro_batch: int = 0) -> Tuple[pd.DataFrame, List[ImageAlphaComparison]]:
        data: Dict[str, List[float]] = {k: [] for k in self.eval_results.keys() if k not in ['X', 'X_tilde']}
        sess = tf.keras.backend.get_session()
        alpha_comparisons: List[ImageAlphaComparison] = []

        for alpha in tqdm(self.bpp_range_adapter.current_alpha_linspace, desc='Alpha grid evaluation, alphas'):
            for _ in trange(self.val_dataset_steps, desc='Batches'):
                # comparison_ids = None
                # alpha_comparisons.extend([ImageAlphaComparison() for _ in range(num_alpha_comparisons_pro_batch)])

                results = sess.run(self.eval_results, {self.alpha_placeholder: alpha})
                for key, collection in data.items():
                    collection.extend(results[key])

                # if comparison_ids is None:
                #     comparison_ids = np.random.choice(len(results['X']), size=num_alpha_comparisons_pro_batch)
                #
                # for i in range(num_alpha_comparisons_pro_batch):
                #     alpha_comparisons[-i].alphas.append(results['alpha'][comparison_ids[i]])
                #     alpha_comparisons[-i].images.append(results['X_tilde'][comparison_ids[i]])
                #     alpha_comparisons[-i].original_image = results['X'][comparison_ids[i]]

        df = pd.DataFrame(data).groupby('alpha').mean().reset_index()
        return df, alpha_comparisons


def area_under_bpp_metric(bpps: np.array, metrics: np.array, bpp_range: Tuple[float, float]) -> float:
    bpps_within_range = (bpps >= bpp_range[0]) & (bpps <= bpp_range[1])
    bpps_filtered = bpps[bpps_within_range]
    metrics_filtered = metrics[bpps_within_range]

    bpps_ordered = bpps_filtered[np.argsort(bpps_filtered)]
    metrics_ordered = metrics_filtered[np.argsort(bpps_filtered)]

    if len(bpps_ordered) < 2:
        return 0.

    return auc(bpps_ordered, metrics_ordered)
