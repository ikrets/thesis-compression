import tensorflow as tf
import numpy as np
from pathlib import Path

MEAN = (0.4914, 0.4822, 0.4465)
SCALE = (0.2023, 0.1994, 0.2010)


def normalize(X, inverse=False):
    if not inverse:
        return (X - MEAN) / SCALE
    else:
        return X * SCALE + MEAN


def filename_to_label(filenames):
    y = np.zeros((len(filenames), 10), dtype=np.uint8)
    for i, file in enumerate(filenames):
        y[i, int(Path(file).parts[-2])] = 1

    return y


def pipeline(filenames, batch_size, flip, crop, classifier_normalize=True, shuffle=True, correct_bgr=False,
             repeat=True,
             num_parallel_calls=8):
    if shuffle:
        perm = np.random.permutation(len(filenames))
        filenames = np.array(filenames)[perm]

    y = filename_to_label(filenames)

    data_X = tf.data.Dataset.from_tensor_slices([str(f) for f in filenames])
    data_X = data_X.map(tf.io.read_file).map(lambda fn: tf.cast(tf.io.decode_png(fn), dtype=tf.float32) / 255.,
                                             num_parallel_calls=8)
    data_X = data_X.map(lambda X: tf.reshape(X, [32, 32, 3]), num_parallel_calls=8)
    if classifier_normalize:
        data_X = data_X.map(lambda X: normalize(X[..., ::-1] if correct_bgr else X), num_parallel_calls=8)

    if crop:
        data_X = data_X.map(lambda X: tf.image.pad_to_bounding_box(X, 4, 4, 40, 40), num_parallel_calls)
        data_X = data_X.map(lambda X: tf.image.random_crop(X, size=[32, 32, 3]), num_parallel_calls)
    if flip:
        data_X = data_X.map(tf.image.random_flip_left_right)

    data_y = tf.data.Dataset.from_tensor_slices(y)
    data = tf.data.Dataset.zip((data_X, data_y))
    if shuffle:
        data = data.shuffle(100)
    data = data.batch(batch_size)

    if repeat:
        data = data.repeat()

    return data.prefetch(tf.data.experimental.AUTOTUNE)
