from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, UpSampling2D
from utils import IMAGE_SIZE, get_mean_std


def get_encoder():
    vgg19 = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    vgg19.trainable = False
    mini_vgg19 = Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    inputs = layers.Input([*IMAGE_SIZE, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return Model(inputs, mini_vgg19_out, name="mini_vgg19")


def get_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = Sequential(
        [
            InputLayer((None, None, 512)),
            Conv2D(filters=512, **config),
            UpSampling2D(),
            Conv2D(filters=256, **config),
            Conv2D(filters=256, **config),
            Conv2D(filters=256, **config),
            Conv2D(filters=256, **config),
            UpSampling2D(),
            Conv2D(filters=128, **config),
            Conv2D(filters=128, **config),
            UpSampling2D(),
            Conv2D(filters=64, **config),
            Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="sigmoid",
            ),
        ]
    )
    return decoder


def ada_in(style, content):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    return t
