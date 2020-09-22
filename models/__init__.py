import tensorflow as tf
import numpy as np

K = tf.keras.backend

def surgery_flatten(model, backbone_prefix):
    for backbone_index, backbone in enumerate(model.layers):
        if backbone.name == backbone_prefix:
            break

    configs = []
    types = []
    weights = []

    old_layers_together = model.layers[:backbone_index] + model.get_layer(backbone_prefix).layers[1:] + model.layers[
                                                                                                        backbone_index + 1:]
    for l in old_layers_together:
        configs.append(l.get_config())
        types.append(type(l))
        weights.append(l.get_weights())

    K.clear_session()
    new_model = tf.keras.Sequential()
    for config, t in zip(configs, types):
        l = t.from_config(config)
        l._name = config['name']
        new_model.add(l)

    new_model.predict(np.random.rand(1, *model.input.shape[1:]))

    for l, w in zip(new_model.layers, weights[1:]):
        l.set_weights(w)

    return new_model
