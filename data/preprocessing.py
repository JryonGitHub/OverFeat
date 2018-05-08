import tensorflow as tf
import random
from data.tf_parser import parse_proto


def rand_contrast(image):
    return tf.image.random_contrast(image, lower=0.5, upper=1.5)


def rand_saturation(image):
    return tf.image.random_saturation(image, lower=0.5, upper=1.5)


def rand_hue(image):
    return tf.image.random_hue(image, max_delta=0.2)


def rand_brightness(image):
    return tf.image.random_brightness(image, max_delta=32. / 255.)


def coloraug_fn(image):
    functions = [rand_brightness, rand_hue, rand_saturation, rand_contrast]
    random.shuffle(functions)
    for func in functions:
        image = func(image)
    return image


def normalized_image(image):
    # Rescale from [0, 255] to [0, 2]
    image = tf.multiply(image, 1. / 127.5)
    # Rescale to [-1, 1]
    return tf.subtract(image, 1.0)


def resize_small(image, height, width):
    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)
    aspect = tf.divide(height, width)
    newdim = tf.cond(tf.less_equal(height, width), lambda: (tf.constant(256, dtype=tf.int32),
                                                            tf.cast(tf.floor(tf.divide(256.0, aspect)),
                                                                    dtype=tf.int32)),
                     lambda: (
                         tf.cast(tf.floor(tf.multiply(256., aspect)), dtype=tf.int32),
                         tf.constant(256, dtype=tf.int32)))

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bicubic(image, newdim)
    image = tf.squeeze(image)
    return image


def central_crop(image):
    imgsize = tf.shape(image)
    total_crop_height = imgsize[0] - 221
    crop_top = total_crop_height // 2
    total_crop_width = imgsize[1] - 221
    crop_left = total_crop_width // 2
    crop = tf.slice(image, [crop_top, crop_left, 0], [221, 221, 3])
    return crop


def preprocess_train(data):
    with tf.device('/cpu:0'):
        data = parse_proto(data)
        image = data['image']
        height = data['height']
        width = data['width']
        label = data['label']
        image = tf.image.decode_jpeg(image, dct_method='INTEGER_ACCURATE')
        image = tf.cast(image, dtype=tf.float32)
        image = tf.reshape(image, tf.stack([height, width, 3]))
        image = resize_small(image, height, width)
        image = tf.random_crop(image, tf.constant([221, 221, 3], dtype=tf.int32))
        image = tf.image.random_flip_left_right(image)
        image /= 255.0
        image = coloraug_fn(image)
        image *= 255.0
        image = normalized_image(image)
        image = tf.transpose(image, [2, 0, 1])
    return image, label


def preprocess_val(data):
    with tf.device('/cpu:0'):
        data = parse_proto(data)
        image = data['image']
        height = data['height']
        width = data['width']
        label = data['label']
        image = tf.image.decode_jpeg(image, dct_method='INTEGER_ACCURATE')
        image = tf.cast(image, dtype=tf.float32)
        image = tf.reshape(image, tf.stack([height, width, 3]))
        image = resize_small(image, height, width)
        image = central_crop(image)
        image = normalized_image(image)
        image = tf.transpose(image, [2, 0, 1])
    return image, label
