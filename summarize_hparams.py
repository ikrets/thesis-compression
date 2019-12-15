import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_dir', type=str, required=True)
parser.add_argument('--kernel_size', type=int, required=True)
parser.add_argument('--scale_hyperprior', action='store_true')
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--hyperprior_steps', type=int)
parser.add_argument('--last_act_mistake', action='store_true')
args = parser.parse_args()

experiments = Path(args.experiments_dir)
all_parameters = experiments.glob('**/parameters.json')

relevant_hparams = {
    'lmbda': 'lmbda',
    'perceptual_loss_layer': 'downstream_loss_layer',
    'perceptual_loss_alpha': 'downstream_loss_alpha'}

sess = tf.Session()

for parameter in all_parameters:
    with parameter.open('r') as fp:
        parameter_dict = json.load(fp)

    experiment_dir = parameter.parent
    hparams = {new_name: parameter_dict[old_name] for old_name, new_name in relevant_hparams.items() if
               old_name in parameter_dict}
    if 'downstream_loss_type' not in hparams:
        hparams['downstream_loss_type'] = 'activation_mse'

    hparams.update({
        'kernel_size': args.kernel_size,
        'scale_hyperprior': args.scale_hyperprior,
        'steps': args.steps
    })

    if args.scale_hyperprior:
        hparams['hyperprior_steps'] = args.hyperprior_steps

    with tf.compat.v2.summary.create_file_writer(str(experiment_dir)).as_default() as w:
        sess.run(w.init())
        sess.run(hp.hparams(hparams))
        sess.run(w.flush())