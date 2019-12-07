import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import math
from tqdm import trange, tqdm

tfk = tf.keras
tfkl = tf.keras.layers

tfd = tfp.distributions
tfpl = tfp.layers


def train_vae(prior, encoder, decoder, optimizer,
              epochs,
              beta_schedule,
              data_train,
              batch_size,
              x_domain,
              z_domain):
    trainable_variables = None

    logs = []
    prior_z = prior.log_prob(tf.tile(z_domain, [1, len(x_domain)]))
    dx = (np.max(x_domain) - np.min(x_domain)) / len(x_domain)

    @tf.function
    def calc_log():
        qz = encoder(tf.tile(data_train[..., np.newaxis], [1, len(z_domain), 1]))
        qz_mean = qz.mean()[:, 0, :]
        qz_std = qz.stddev()[:, 0, :]
        qz = qz.prob(
            tf.tile(z_domain[np.newaxis, ...].astype(np.float32), multiples=[len(data_train), 1, 1]))
        qz = (tf.reduce_sum(qz, axis=[0]) / len(data_train))

        px = decoder(tf.tile(z_domain[..., np.newaxis], [1, len(x_domain), 1]))
        px_mean = px.mean()[:, 0, :]
        px_std = px.stddev()[:, 0, :]
        px = px.log_prob(
            tf.tile(x_domain[np.newaxis, ...].astype(np.float32), multiples=[len(z_domain), 1, 1]))
        px = tf.exp(px + prior_z)
        px = (tf.reduce_sum(px, axis=[0]) * dx)

        return {'qz': qz, 'px': px, 'qz_mean': qz_mean, 'qz_std': qz_std,
                'px_mean': px_mean, 'px_std': px_std}

    @tf.function
    def forward(x, beta):
        q_z_x = encoder(x)
        z_x = q_z_x.sample()
        kld = tf.reduce_mean(q_z_x.log_prob(z_x) - prior.log_prob(z_x))
        rv_x = decoder(z_x)

        nll = tf.reduce_mean(-rv_x.log_prob(x))
        elbo = nll + beta * kld

        return kld, nll, elbo

    step = 0
    for _ in trange(epochs, desc='epoch'):
        for _ in trange(math.ceil(len(data_train) // batch_size), desc='step'):
            x = data_train[np.random.choice(len(data_train), size=batch_size)]

            with tf.GradientTape() as tape:
                kld, nll, elbo = forward(x, beta_schedule(step))

            log_before = {'before_' + k: x.numpy() for k, x in calc_log().items()}

            if trainable_variables is None:
                trainable_variables = encoder.trainable_variables + decoder.trainable_variables
                if hasattr(prior, 'trainable_variables'):
                    trainable_variables += prior.trainable_variables

            gradients = tape.gradient(elbo, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            log_after = {'after_' + k: x.numpy() for k, x in calc_log().items()}

            log = {'step': step, 'kld': kld.numpy(), 'nll': nll.numpy(), 'elbo': elbo.numpy()}
            log.update(log_before)
            log.update(log_after)

            logs.append(log)

            step += 1

    return logs


if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], enable=True)

    np.random.seed(1)
    train_len = 600
    batch_size = 64
    data_train = np.concatenate(
        [np.random.normal(loc=-0.5, scale=0.5, size=train_len // 2),
         np.random.normal(loc=1, scale=0.05, size=train_len // 2)]).astype(
        np.float32)
    data_train = data_train[:, np.newaxis]

    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[1]),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(tfpl.IndependentNormal.params_size(event_shape=[1]), activation=None),
        tfpl.IndependentNormal(event_shape=[1])
    ])

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[1]),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(tfpl.IndependentNormal.params_size(event_shape=[1]), activation=None),
        tfpl.IndependentNormal(event_shape=[1])
    ])

    prior = tfd.Normal(loc=0, scale=1)

    beta_schedule = lambda step: min(step / 30, 1)
    optimizer = tfk.optimizers.Adam(5e-4)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    logs = train_vae(prior, encoder, decoder, optimizer,
                     epochs=100,
                     data_train=data_train,
                     batch_size=64,
                     beta_schedule=beta_schedule,
                     x_domain=np.linspace(-6, 6, 200)[:, np.newaxis].astype(np.float32),
                     z_domain=np.linspace(-6, 6, 200)[:, np.newaxis].astype(np.float32))

    with open('result.pickle', 'wb') as fp:
        pickle.dump(logs, fp)
