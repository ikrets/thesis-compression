import dataclasses
import tensorflow as tf


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
