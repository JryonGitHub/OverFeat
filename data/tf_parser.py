import tensorflow as tf


def parse_proto(buffer):
    with tf.device('/cpu:0'):
        feature_map = {
            'image': tf.FixedLenFeature([], tf.string, default_value=''),
            'height': tf.FixedLenFeature([], tf.int64, default_value=-1),
            'width': tf.FixedLenFeature([], tf.int64, default_value=-1),
            'label': tf.FixedLenFeature([], tf.int64, default_value=-1)
        }
        features = tf.parse_single_example(buffer, feature_map)
    return features
