import tensorflow as tf
import numpy as np
import cv2
from queue import Queue
from threading import Thread
from pathlib import Path
from tqdm import trange

from datasets import cifar10
from experiment import save_experiment_params
import models

K = tf.keras.backend
AUTO = tf.data.experimental.AUTOTUNE


# taken from https://github.com/eclique/keras-gradcam
def grad_cam_batch(input_model, batch, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    layer_output = input_model.get_layer(layer_name).output
    loss = tf.gather_nd(input_model.output, np.dstack([range(batch['X'].shape[0]), batch['label']])[0])

    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([batch['X'], 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)

    # Process CAMs
    new_cams = np.empty((batch['X'].shape[:3]))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, batch['X'].shape[1:3], cv2.INTER_CUBIC)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


def writer_fn(output_dir: Path, queue: Queue) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        heatmap_int = (item['gradcam_heatmap'] * 255).astype(np.uint8)

        for i in range(item['name'].shape[0]):
            target_file = output_dir / item['name'][i].decode('ascii')
            target_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target_file), heatmap_int[i])


def save_grad_cam_outputs(dataset, dataset_examples, model, layer_name, output_dir):
    sess = K.get_session()
    it = dataset.make_one_shot_iterator()
    next = it.get_next()

    results_queue: Queue = Queue(maxsize=20)
    writer_thread = Thread(target=writer_fn, args=(output_dir, results_queue))
    writer_thread.start()

    for _ in trange(dataset_examples):
        item = sess.run(next)
        heatmap = grad_cam_batch(model, item, layer_name)
        results_queue.put_nowait({'name': item['name'],
                                  'gradcam_heatmap': heatmap})
    results_queue.put_nowait(None)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--model_correct_bgr', action='store_true')
    parser.add_argument('--model_backbone_prefix', type=str)
    parser.add_argument('--gradcam_layer_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    if args.model_weights:
        model.load_weights(args.model_weights)
    if args.model_backbone_prefix:
        model = models.surgery_flatten(model, args.model_backbone_prefix)

    data_train, train_examples = cifar10.read_images(Path(args.dataset) / 'train')
    data_test, test_examples = cifar10.read_images(Path(args.dataset) / 'test')


    def preprocess_cifar10(item):
        img = tf.io.decode_png(item['X'])
        img = tf.reshape(img, shape=[32, 32, 3])
        img = tf.cast(img, dtype=tf.float32) / 255.
        img = cifar10.normalize(img)
        if args.model_correct_bgr:
            img = img[..., ::-1]

        return {'name': item['name'], 'X': img, 'label': cifar10.filename_to_label(item['name'])}


    data_train = data_train.map(preprocess_cifar10, AUTO).batch(args.batch_size).prefetch(AUTO)
    data_test = data_test.map(preprocess_cifar10, AUTO).batch(args.batch_size).prefetch(AUTO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_params(output_dir, args)

    save_grad_cam_outputs(data_train, np.ceil(train_examples / args.batch_size).astype(int), model,
                          args.gradcam_layer_name,
                          output_dir)
    save_grad_cam_outputs(data_test, np.ceil(test_examples / args.batch_size).astype(int), model,
                          args.gradcam_layer_name,
                          output_dir)
