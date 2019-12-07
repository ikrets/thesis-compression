import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


def block(inp, filters, stride):
    out = tfkl.Conv2D(filters=filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(inp)
    out = tfkl.BatchNormalization()(out)
    out = tfkl.Activation('relu')(out)
    out = tfkl.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False)(out)
    out = tfkl.BatchNormalization()(out)

    if stride != 1:
        shortcut = tfkl.Conv2D(filters=filters, kernel_size=1, strides=stride, use_bias=False)(inp)
        shortcut = tfkl.BatchNormalization()(shortcut)
    else:
        shortcut = inp

    out = tfkl.Add()([out, shortcut])
    out = tfkl.Activation('relu')(out)

    return out

def resnet18(inp):
    def repeat_block(inp, filters, stride, num_blocks):
        out = inp
        for i in range(num_blocks):
            out = block(out, filters, stride=stride if i == 0 else 1)

        return out

    out = tfkl.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(inp)
    out = tfkl.BatchNormalization()(out)
    out = repeat_block(out, 64, num_blocks=2, stride=1)
    out = repeat_block(out, 128, num_blocks=2, stride=2)
    out = repeat_block(out, 256, num_blocks=2, stride=2)
    out = repeat_block(out, 512, num_blocks=2, stride=2)

    out = tfkl.AvgPool2D(4)(out)
    out = tfkl.Flatten()(out)
    out = tfkl.Dense(10, activation='softmax')(out)

    return tfk.Model(inputs=inp, outputs=out)

