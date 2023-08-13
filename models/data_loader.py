import os
import tensorflow as tf
from .utils import decode_and_resize, IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32


def create_image_loader(path):
    images = os.listdir(path)
    images = [os.path.join(path, p) for p in images]

    # split the images in train, val and test
    total_images = len(images)
    train = images[: int(0.8 * total_images)]
    val = images[int(0.8 * total_images) : int(0.9 * total_images)]
    test = images[int(0.9 * total_images) :]

    # Build the tf.data datasets.
    train_ds = (
        tf.data.Dataset.from_tensor_slices(train)
        .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        .repeat()
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val)
        .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        .repeat()
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test)
        .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        .repeat()
    )

    return train_ds, val_ds, test_ds


def data_loader(style_path="../data/art_style", content_path="../data/ffhq-256/"):
    train_style_ds, val_style_ds, test_style_ds = create_image_loader(style_path)
    train_content_ds, val_content_ds, test_content_ds = create_image_loader(content_path)

    # Zipping the style and content datasets.
    train_ds = (
        tf.data.Dataset.zip((train_style_ds, train_content_ds))
        .shuffle(BATCH_SIZE * 2)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.zip((val_style_ds, val_content_ds))
        .shuffle(BATCH_SIZE * 2)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.zip((test_style_ds, test_content_ds))
        .shuffle(BATCH_SIZE * 2)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds, test_ds