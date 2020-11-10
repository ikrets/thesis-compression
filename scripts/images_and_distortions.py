import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

from datasets.imagenette import read_compressed_tfrecords, normalize
from experiment import save_experiment_params

AUTO = tf.data.experimental.AUTOTUNE
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--original_dataset', type=str, required=True)
parser.add_argument('--original_model', type=str, required=True)
parser.add_argument('--original_model_readout', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_names', type=str, nargs='+', required=True)
parser.add_argument('--compressed_datasets', type=str, required=True)
parser.add_argument('--bpg_datasets', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
save_experiment_params(args.output_dir, args)

model = tf.keras.models.load_model(args.original_model)
output = model.get_layer(args.original_model_readout).output
readout_model = tf.keras.Model(inputs=model.input, outputs=output)

image_names = set(args.image_names)
original_dataset = Path(args.original_dataset)

compressed_datasets = Path(args.compressed_datasets).glob('**/compressed.tfrecord')

bpg_files = Path(args.bpg_datasets).glob('**/val/*/*.png')
bpg_files = [p for p in bpg_files if p.name in image_names]


def filter_images(name):
    file_name = name.decode('ascii').split('/')[-1]
    return file_name in image_names


def collect_bpg_bpp(name):
    name_bpg = tf.strings.regex_replace(name, '.png', '.bpg')
    bpg_string = tf.io.read_file(name_bpg)
    bpp = (tf.strings.length(bpg_string) - 13) * 8 / 256 / 256
    return bpp


def output_filename_bpg(name, bpp):
    name = name.decode('ascii')
    qp_value = Path(name).parts[-4]
    file_name = 'bpg_{}_{:0.2f}.png'.format(qp_value, bpp)
    output_path = output_dir / Path(name).stem / file_name
    return str(output_path)


def save_file(input_filename, output_filename):
    tf.io.write_file(output_filename, tf.io.read_file(input_filename))
    return 0


bpg_data = tf.data.Dataset.from_tensor_slices([str(p) for p in bpg_files])
bpg_data = bpg_data.map(lambda fn: (fn, collect_bpg_bpp(fn)), AUTO)
bpg_data = bpg_data.map(lambda fn, bpp: (fn, tf.py_func(output_filename_bpg,
                                                        [fn, bpp],
                                                        tf.string
                                                        )), AUTO)
bpg_data = bpg_data.map(save_file, AUTO)
bpg_data.reduce(np.int32(0), lambda acc, item: 0)

for dataset in compressed_datasets:
    compressor, compressor_param = dataset.parts[-4:-2]
    _, data = read_compressed_tfrecords([dataset])


    def output_filename(name, bpp):
        file_name = '{}_{}_{:0.2f}.png'.format(compressor, compressor_param, bpp)
        output_path = output_dir / Path(name.decode('ascii')).stem / file_name
        return str(output_path)


    def write_file(X_reconstructed, output_filename):
        tf.io.write_file(output_filename, X_reconstructed)
        return 0


    data = data.filter(lambda item: tf.py_func(filter_images, [item['name']], tf.bool))
    data = data.map(lambda item: (item['X'],
                                  tf.py_func(output_filename, [item['name'], item['range_coded_bpp']], tf.string)),
                    AUTO)
    data = data.map(write_file, AUTO)
    data.reduce(np.int32(0), lambda acc, item: 0)

for image_name in image_names:
    image_class = image_name.split('_')[0]
    image_name = Path(image_name)
    original_img = np.array(Image.open(original_dataset / 'val' / image_class / image_name))
    original_img = original_img / 255

    original_prediction = readout_model.predict(normalize(original_img[np.newaxis, ...]))


    def write_mse(reconstruction_img, name):
        mse_img = tf.reduce_mean(tf.math.squared_difference(reconstruction_img, original_img), axis=-1, keepdims=True)
        mse_img = tf.saturate_cast(mse_img * 2 ** 16, tf.uint16)
        mse_img_name = tf.strings.regex_replace(name, '.png', '.mse.png')
        tf.io.write_file(mse_img_name, tf.image.encode_png(mse_img))
        return 0


    reconstruction_names = [str(p) for p in (output_dir / image_name.stem).glob('*.png')]
    reconstruction_names = tf.data.Dataset.from_tensor_slices(reconstruction_names)
    reconstructions_img = reconstruction_names.map(tf.io.read_file).map(tf.image.decode_png, AUTO)
    reconstructions_img = reconstructions_img.map(lambda img: tf.cast(img, tf.float32) / 255, AUTO)
    reconstructions = tf.data.Dataset.zip((reconstructions_img, reconstruction_names))
    reconstructions = reconstructions.map(write_mse, AUTO)
    reconstructions.reduce(np.int32(0), lambda acc, item: 0)

    prediction_extension = '_{}.mse.png'.format(args.original_model_readout)


    def write_layer_mse(prediction, name):
        mse_img = tf.reduce_max(tf.math.squared_difference(prediction, original_prediction), axis=-1, keepdims=True)
        mse_img = tf.saturate_cast(mse_img * 2 ** 16, tf.uint16)
        mse_img_name = tf.strings.regex_replace(name, '.png', prediction_extension)
        tf.io.write_file(mse_img_name, tf.image.encode_png(mse_img[0]))
        return 0


    reconstructions_img_normalized = reconstructions_img.map(normalize, AUTO).batch(args.batch_size).prefetch(AUTO)
    predictions = readout_model.predict(reconstructions_img_normalized)
    predictions = tf.data.Dataset.from_tensor_slices(predictions)
    predictions = tf.data.Dataset.zip((predictions, reconstruction_names))
    predictions = predictions.map(write_layer_mse, AUTO)
    predictions.reduce(np.int32(0), lambda acc, item: 0)
