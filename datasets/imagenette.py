import tensorflow as tf
import json
from pathlib import Path
from typing import Union, Tuple, Optional

AUTO = tf.data.experimental.AUTOTUNE


def path_to_files_labels(path, extension):
    label_to_id = {l[1].name: l[0] for l in enumerate(sorted(path.glob('*/')))}
    files = list(path.glob('*/*.{}'.format(extension)))
    labels = [label_to_id[t.parent.name] for t in files]
    files = [str(f) for f in files]

    return files, labels


def preprocess_img(image_bytes, size, is_training):
    image = tf.image.decode_jpeg(image_bytes)
    image = tf.cond(tf.equal(tf.shape(image)[-1], 1),
                    true_fn=lambda: tf.image.grayscale_to_rgb(image),
                    false_fn=lambda: image)
    image = tf.cast(image, tf.float32) / 255

    return image

def normalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return (image - mean) / std


def read_images(dir: Union[str, Path], extension='png') -> Tuple[tf.data.Dataset, int]:
    files, labels = path_to_files_labels(Path(dir), extension)
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(files),
                                   tf.data.Dataset.from_tensor_slices(labels)))
    dataset = dataset.map(lambda X_fname, Y: (tf.io.read_file(X_fname), Y))

    return dataset, len(files)


def augment(image, Y, size):
    image = tf.image.pad_to_bounding_box(image, size // 8, size // 8,
                                         size + size // 8, size + size // 8)
    image = tf.image.random_crop(image, size=[size, size, 3])
    image = tf.image.random_flip_left_right(image)
    return image, Y


def imagenette_to_imagenet_mapping():
    with open('datasets/imagenet_class_index.json', 'r') as fp:
        class_index = json.load(fp)

    return {v[0]: int(k) for k, v in class_index.items()}


def pipeline(dataset, batch_size, size, is_training,
             cache=False,
             repeat=True):
    def preprocess_fn(X, Y):
        return preprocess_img(X, size=size, is_training=is_training), tf.one_hot(Y, depth=10)

    # if not caching, shuffle the filenames
    if not cache and is_training:
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(preprocess_fn, AUTO)
    if cache:
        dataset = dataset.cache()
    # if caching, need to shuffle the images
    if cache and is_training:
        dataset = dataset.shuffle(10000)

    if is_training:
        def augment_fn(image, Y):
            return augment(image, Y, size)

        dataset = dataset.map(augment_fn, AUTO)

    dataset = dataset.map(lambda X, Y: {'X': X, 'label': Y}, AUTO)
    dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    return dataset
