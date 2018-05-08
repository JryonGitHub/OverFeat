import numpy as np
import tensorflow as tf

from networks.overfeataccuratebase import OverFeatAccurateBase

inp = np.random.randn(10, 221, 221, 3)

input = tf.placeholder(dtype=tf.float32, shape=(None, 221, 221, 3),
                       name='input')

mode_train = tf.constant(True)

mode_val = tf.constant(False)

net = OverFeatAccurateBase(input, 1000)

logits = net.logits

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./tboard', graph=sess.graph)
    sess.run(init_op)
    print(sess.run(logits, feed_dict={input: inp, net.mode:False}))
    writer.close()
