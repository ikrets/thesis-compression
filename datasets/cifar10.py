import tensorflow as tf
from pathlib import Path
from typing import Optional, Union, Tuple, Sequence

import datasets.compressed

AUTO = tf.data.experimental.AUTOTUNE

MEAN = (0.4914, 0.4822, 0.4465)
SCALE = (0.2023, 0.1994, 0.2010)


def normalize(X, inverse=False):
    if not inverse:
        return (X - MEAN) / SCALE
    else:
        return X * SCALE + MEAN


def filename_to_one_hot_label(fname):
    parts = tf.strings.split([fname], sep='/').values
    cl = tf.strings.to_number(parts[-2], out_type=tf.int32)
    return tf.one_hot(cl, depth=10)


def process_image(img_string: tf.Tensor) -> tf.Tensor:
    img = tf.reshape(tf.io.decode_png(img_string, channels=3), (32, 32, 3))
    img = tf.cast(img, tf.float32) / 255.
    return img


def read_images(dir: Union[str, Path]) -> Tuple[tf.data.Dataset, int]:
    files = tf.io.gfile.glob(f'{dir}/*/*/*.png')
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(lambda fname: {'X': tf.io.read_file(fname),
                                         'name': fname})

    return dataset, len(files)


def read_compressed_tfrecords(files: Sequence[Union[str, Path]]) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset = tf.data.TFRecordDataset([files])

    dataset = dataset.map(datasets.compressed.deserialize_example, AUTO)
    train_dataset = dataset.filter(lambda item: tf.strings.regex_full_match(item['name'], '.*train.*'))
    test_dataset = dataset.filter(lambda item: tf.strings.regex_full_match(item['name'], '.*test.*'))

    return train_dataset, test_dataset


def pipeline(dataset, batch_size, flip, crop, classifier_normalize=True,
             shuffle_buffer_size: Optional[int] = None,
             correct_bgr=False,
             repeat=True):
    dataset = dataset.map(lambda item: {'X': process_image(item['X']),
                                        'name': item['name'],
                                        'label': filename_to_one_hot_label(item['name'])},
                          AUTO)

    def process_X(X):
        if classifier_normalize:
            X = normalize(X[..., ::-1] if correct_bgr else X)
        if crop:
            X = tf.image.pad_to_bounding_box(X, 4, 4, 40, 40)
            X = tf.image.random_crop(X, size=[32, 32, 3])
        if flip:
            X = tf.image.random_flip_left_right(X)
        return X

    dataset = dataset.map(lambda item: (process_X(item['X']), item['label']))
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    return dataset
