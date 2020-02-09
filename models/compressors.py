import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from models.layers import FiLM

tfk = tf.keras
tfkl = tf.keras.layers


class AnalysisTransform(tfkl.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, use_FiLM, FiLM_depth=None, FiLM_width=None, FiLM_activation=None, *args, **kwargs):
        super(AnalysisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.use_FiLM = use_FiLM
        if self.use_FiLM:
            if not FiLM_width or not FiLM_depth or not FiLM_activation:
                raise ValueError('If use_FiLM is set, other FiLM parameters must be set too!')

            self.FiLM_width = FiLM_width
            self.FiLM_depth = FiLM_depth
            self.FiLM_activation = FiLM_activation

    def build(self, input_shape):
        if self.use_FiLM:
            parameters_input = tfkl.Input(batch_shape=input_shape[0])
            image_input = tfkl.Input(batch_shape=input_shape[1])
            inputs = [parameters_input, image_input]
        else:
            image_input = tfkl.Input(input_shape)
            inputs = image_input

        net = image_input
        for i in range(4):
            net = tfc.SignalConv2D(
                self.num_filters, (5, 5), name=f"layer_{i}", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name=f"gdn_{i}"))(net)

            if self.use_FiLM:
                net = FiLM(width=self.FiLM_width, depth=self.FiLM_depth,
                           activation=self.FiLM_activation if i != 3 else None,
                           name=f'layer_{i}_film')([parameters_input, net])

        self.net = tfk.Model(inputs=inputs, outputs=net)

    def call(self, input_tensor, **kwargs):
        return self.net(input_tensor)


class SynthesisTransform(tfkl.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, use_FiLM, FiLM_depth=None, FiLM_width=None, FiLM_activation=None, *args, **kwargs):
        super(SynthesisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.use_FiLM = use_FiLM
        if self.use_FiLM:
            if not FiLM_width or not FiLM_depth or not FiLM_activation:
                raise ValueError('If use_FiLM is set, other FiLM parameters must be set too!')

            self.FiLM_width = FiLM_width
            self.FiLM_depth = FiLM_depth
            self.FiLM_activation = FiLM_activation

    def build(self, input_shape):
        if self.use_FiLM:
            parameters_input = tfkl.Input(batch_shape=input_shape[0])
            bottleneck_input = tfkl.Input(batch_shape=input_shape[1])
            inputs = [parameters_input, bottleneck_input]
        else:
            bottleneck_input = tfkl.Input(batch_shape=input_shape)
            inputs = bottleneck_input

        net = bottleneck_input
        for i in range(3):
            net = tfc.SignalConv2D(
                self.num_filters, (5, 5), name=f"layer_{i}", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name=f"igdn_{i}", inverse=True))(net)
            if self.use_FiLM:
                net = FiLM(width=self.FiLM_width, depth=self.FiLM_depth, activation=self.FiLM_activation,
                           name=f'layer_{i}_film')([parameters_input, net])

        net = tfc.SignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None)(net)
        if self.use_FiLM:
            net = FiLM(width=self.FiLM_width, depth=self.FiLM_depth, activation=None, name='layer_3_film')(
                [parameters_input, net])

        self.net = tfk.Model(inputs=inputs, outputs=net)

    def call(self, input_tensor, **kwargs):
        return self.net(input_tensor)
