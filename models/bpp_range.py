import tensorflow as tf
import pandas as pd
import numpy as np
from models.compressors import SimpleFiLMCompressor, bits_per_pixel, pipeline_add_constant_parameters
from scipy.optimize import curve_fit
from sklearn.metrics import auc
from tqdm import tqdm, trange
from typing import Tuple, Sequence, Dict, List

from models.downstream_losses import PerceptualLoss


def log_curve(x, a, b, c, d):
    return a * np.log(b * x + c) + d


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
        if np.max(bpps) - np.min(bpps) < .2:
            raise RuntimeError('The bpps are too close, skipping fit.')

        popt, _ = curve_fit(log_curve, alphas, bpps)
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
        self._update_current_alpha_linspace()

        batch = tf.compat.v1.data.make_one_shot_iterator(self.eval_dataset).get_next()
        batch = pipeline_add_constant_parameters(batch, self._alpha, self.lmbda)
        compressor_output = self.compressor.forward(batch, training=False)
        self.bpp = bits_per_pixel(compressor_output['Y_likelihoods'], batch['X'].shape)

        sess = tf.keras.backend.get_session()
        sess.run(tf.variables_initializer([self._alpha]))

    def _update_current_alpha_linspace(self) -> None:
        self.current_alpha_linspace = self.bpp_to_alpha.numpy_call(self.bpp_linspace)

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
        self._update_current_alpha_linspace()

    def add_computed_parameters(self, dataset_item: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(dataset_item['X'])[0]
        random_bpps = tf.random.uniform((batch_size,), self.bpp_range[0], self.bpp_range[1])
        random_alphas = self.bpp_to_alpha(random_bpps)

        dataset_item.update({
            'lambda': tf.repeat(self.lmbda, batch_size),
            'alpha': tf.maximum(0., random_alphas)
        })

        return dataset_item


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

    class ImageAlphaComparison:
        alphas: List[float] = []
        images: List[np.array] = []
        original_image: np.array

    def evaluate(self, num_alpha_comparisons_pro_batch: int = 0) -> Tuple[pd.DataFrame, List[ImageAlphaComparison]]:
        data: Dict[str, List[float]] = {k: [] for k in self.eval_results.keys() if k not in ['X', 'X_tilde']}
        sess = tf.keras.backend.get_session()
        alpha_comparisons: List[BppRangeEvaluation.ImageAlphaComparison] = []

        for alpha in tqdm(self.bpp_range_adapter.current_alpha_linspace, desc='Alpha grid evaluation, alphas'):
            for _ in trange(self.val_dataset_steps, desc='Batches'):
                # comparison_ids = None
                # alpha_comparisons.extend([self.ImageAlphaComparison() for _ in range(num_alpha_comparisons_pro_batch)])

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

def area_under_bpp_metric(evaluation_df: pd.DataFrame, bpp_range: Tuple[float, float]) -> float:
    try:
        popt, _ = curve_fit(log_curve, evaluation_df['bpp'], evaluation_df['downstream_metric'])
        a, b, c, d = popt

        F = lambda x: (b * x + c) * (a * np.log(b * x + c) - a + d)
        return F(bpp_range[1]) - F(bpp_range[0])
    except RuntimeError:
        return 0.
