import tensorflow as tf

tfkl = tf.keras.layers


def vgg16(inp):
    vgg16_backbone = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                       weights=None,
                                                       input_shape=(32, 32, 3))
    out = vgg16_backbone(inp)
    out = tfkl.Flatten()(out)
    out = tfkl.Dense(10, activation='softmax')(out)

    return tf.keras.Model(inputs=inp, outputs=out)
