import tensorflow as tf

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