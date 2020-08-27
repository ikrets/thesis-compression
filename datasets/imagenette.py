import tensorflow as tf
from pathlib import Path
from typing import Union, Tuple, Optional

AUTO = tf.data.experimental.AUTOTUNE


def path_to_files_labels(path):
    label_to_id = {l[1].name: l[0] for l in enumerate(sorted(path.glob('*/')))}
    files = list(path.glob('*/*.JPEG'))
    labels = [label_to_id[t.parent.name] for t in files]
    files = [str(f) for f in files]

    return files, labels


def preprocess_img(image_bytes):
    image = tf.image.decode_jpeg(image_bytes)
    image = tf.cond(tf.equal(tf.shape(image)[-1], 1),
                    true_fn=lambda: tf.image.grayscale_to_rgb(image),
                    false_fn=lambda: image)
    image = tf.cast(image, tf.float32) / 255
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return image


def read_images(dir: Union[str, Path]) -> Tuple[tf.data.Dataset, int]:
    files, labels = path_to_files_labels(Path(dir))
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(files),
                                   tf.data.Dataset.from_tensor_slices(labels)))
    dataset = dataset.map(lambda X_fname, Y: (tf.io.read_file(X_fname), Y))

    return dataset, len(files)


def augment(image, Y, size):
    random_scale = tf.random.uniform([1], minval=1, maxval=1.25)[0]
    image = tf.compat.v2.image.resize_with_pad(image,
                                               tf.cast(size * random_scale, tf.int32),
                                               tf.cast(size * random_scale, tf.int32),
                                               method='bicubic'
                                               )
    image = tf.image.random_crop(image, [size, size, 3])
    image = tf.compat.v2.image.resize_with_pad(image, size, size, method='bicubic')
    image = tf.image.random_flip_left_right(image)
    return image, Y


def pipeline(dataset, batch_size, size, is_training,
             min_height, min_width,
             repeat=True):
    if is_training:
        dataset = dataset.shuffle(10000)

    def preprocess_fn(X, Y):
        return preprocess_img(X), tf.one_hot(Y, depth=10)

    dataset = dataset.map(preprocess_fn, AUTO)

    def filter_fn(X, _):
        return tf.shape(X)[0] > min_height and tf.shape(X)[1] > min_width

    dataset = dataset.filter(filter_fn)

    if is_training:
        def augment_fn(image, Y):
            return augment(image, Y, size)

        dataset = dataset.map(augment_fn, AUTO)
    else:
        def resize_fn(X, Y):
            return tf.compat.v2.image.resize_with_pad(X, size, size, method='bicubic'), Y

        dataset = dataset.map(resize_fn, AUTO)

    dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    return dataset
