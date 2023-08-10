from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from .utils import IMAGE_SIZE


def get_loss_net():
    vgg19 = VGG19(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
    )
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = Model(vgg19.input, outputs)

    inputs = Input([*IMAGE_SIZE, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return Model(inputs, mini_vgg19_out, name="loss_net")
