import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from typing import Union, Tuple, Sequence

import datasets

AUTO = tf.data.experimental.AUTOTUNE


def path_to_files_labels(path, extension):
    label_to_id = {l[1].name: l[0] for l in enumerate(sorted(path.glob('*/')))}
    files = list(path.glob('*/*.{}'.format(extension)))
    labels = [label_to_id[t.parent.name] for t in files]
    files = [str(f) for f in files]

    return files, labels

def get_class_to_label_map(file_dataset_path):
    file_dataset_path = Path(file_dataset_path) / 'train'
    return {l[1].name: l[0] for l in enumerate(sorted(file_dataset_path.glob('*/')))}



def preprocess_img(image_bytes):
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

    def make_relative(fname):
        parts = tf.strings.split([fname], '/')
        return tf.strings.reduce_join(parts.values[-3:], separator='/')

    dataset = dataset.map(lambda fname, Y: {'X': tf.io.read_file(fname),
                                            'name': make_relative(fname),
                                            'label': Y})

    return dataset, len(files)

def read_compressed_tfrecords(files: Sequence[Union[str, Path]]) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset = tf.data.TFRecordDataset([str(f) for f in files])

    dataset = dataset.map(datasets.deserialize_example, AUTO)
    train_dataset = dataset.filter(lambda item: tf.strings.regex_full_match(item['name'], '.*train.*'))
    test_dataset = dataset.filter(lambda item: tf.strings.regex_full_match(item['name'], '.*val.*'))

    return train_dataset, test_dataset


def augment(image, Y):
    size = tf.shape(image)[1]
    image = tf.image.pad_to_bounding_box(image, size // 8, size // 8,
                                         size + size // 8, size + size // 8)
    image = tf.image.random_crop(image, size=[size, size, 3])
    image = tf.image.random_flip_left_right(image)
    return image, Y


def imagenette_to_imagenet_mapping():
    with open('datasets/imagenet_class_index.json', 'r') as fp:
        class_index = json.load(fp)

    return {v[0]: int(k) for k, v in class_index.items()}


def pipeline(dataset, batch_size, is_training,
             cache=False,
             repeat=True):
    def preprocess_fn(item):
        return preprocess_img(item['X']), tf.one_hot(item['label'], depth=10)

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
        dataset = dataset.map(augment, AUTO)

    dataset = dataset.map(lambda X, Y: {'X': X, 'label': Y}, AUTO)
    dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    return dataset

def count_bpg_bpps(dir: Union[str, Path]) -> float:
    def read_bpg_and_png(fname):
        bpg_string = tf.io.read_file(fname)
        png_string = tf.io.read_file(tf.strings.regex_replace(fname, '.bpg$', '.png'))
        return bpg_string, png_string

    files = tf.data.Dataset.list_files(f'{dir}/*/*.bpg')
    file_count = files.reduce(np.float64(0.), lambda acc, _: acc + 1.)
    images = files.map(read_bpg_and_png)

    def reduce_bpp(acc, item):
        bpg_string, png_string = item
        img = tf.image.decode_png(png_string)
        image_bpp = tf.cast(tf.strings.length(bpg_string), tf.float64) * np.float64(8)
        image_bpp /= tf.cast(tf.shape(img)[0] * tf.shape(img)[1], tf.float64)
        return acc + image_bpp / file_count

    return images.reduce(np.float64(0.), reduce_bpp)
