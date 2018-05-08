import numpy as np
import tensorflow as tf
from data.datasets import traindb
import glob
import os

from networks.overfeataccuratebase import OverFeatAccurateBase

#inp = np.random.randn(10, 221, 221, 3)

tfrecorddir = '/data/stars/share/STARSDATASETS/ILSVRC2017/tfrecords/train'

files = list(glob.iglob(os.path.join(tfrecorddir, '*.tfrecords')))

db = traindb(files, batchsize=64, numworkers=1, workerindex=0)

iterator = db.make_initializable_iterator()

iterator_init = iterator.initializer

#input = tf.placeholder(dtype=tf.float32, shape=(None, 221, 221, 3),
                       #name='input')

input = iterator.get_next()

images = input[0]

mode_train = tf.constant(True)

mode_val = tf.constant(False)

net = OverFeatAccurateBase(images, 1000)

logits = net.logits

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./tboard', graph=sess.graph)
    sess.run(init_op)
    sess.run(iterator_init)
    print(sess.run(logits, feed_dict={net.mode:False}))
    writer.close()
