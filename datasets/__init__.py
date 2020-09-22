import dataclasses
import tensorflow as tf
from typing import Dict, Tuple


@dataclasses.dataclass
class DatasetSetup:
    train_dataset: tf.data.Dataset
    train_examples: int
    train_steps: int

    val_dataset: tf.data.Dataset
    val_examples: int
    val_steps: int


def get_mse(item: tf.Tensor, uncompressed_data_dir: str):
    original_file = tf.strings.join([uncompressed_data_dir, item['name']], separator='/')
    original_img = tf.image.decode_png(tf.io.read_file(original_file))
    original_img = tf.cast(original_img, tf.float32) / 255
    compressed_img = tf.cast(tf.image.decode_png(item['X']), tf.float32) / 255
    return tf.reduce_mean(tf.math.squared_difference(original_img, compressed_img), [0, 1, 2])


def serialize_example(name, range_coded_bpp, X, alpha, lmbda):
    feature = {
        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()])),
        'range_coded_bpp': tf.train.Feature(float_list=tf.train.FloatList(value=[range_coded_bpp])),
        'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X])),
        'alpha': tf.train.Feature(float_list=tf.train.FloatList(value=[alpha])),
        'lambda': tf.train.Feature(float_list=tf.train.FloatList(value=[lmbda])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def deserialize_example(example_string):
    feature_description = {
        'name': tf.io.FixedLenFeature([], tf.string),
        'range_coded_bpp': tf.io.FixedLenFeature([], tf.float32),
        'X': tf.io.FixedLenFeature([], tf.string),
        'alpha': tf.io.FixedLenFeature([], tf.float32),
        'lambda': tf.io.FixedLenFeature([], tf.float32)
    }

    return tf.io.parse_single_example(example_string, feature_description)

def dict_to_tuple(item: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    return item['X'], item['label']

