import tensorflow as tf

# Defining the global variables.
IMAGE_SIZE = (224, 224)

EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE


def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation


def decode_and_resize(image_path):
    """Decodes and resizes an image from the image file path.

    Args:
        image_path: The image file path.

    Returns:
        A resized image.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def get_sample_images():
    test_style = decode_and_resize("../data/sample/William_Turner_16.jpg")
    test_content = decode_and_resize("../data/sample/hp2.jpg")

    test_style = tf.expand_dims(test_style, axis=0)
    test_content = tf.expand_dims(test_content, axis=0)

    return test_style, test_content