import tensorflow as tf


class OverFeatAccurateBase(object):
    def __init__(self, minibatch, numclasses):
        self._numclasses = numclasses
        self._trainmode = tf.placeholder(tf.bool)
        self._logits = self._buildmodel(minibatch)

    @property
    def numclasses(self):
        return self._numclasses

    @property
    def mode(self):
        return self._trainmode

    @mode.setter
    def mode(self, val):
        self._trainmode=val

    @property
    def logits(self):
        return self._logits

    def _buildmodel(self, minibatch):
        out = tf.layers.conv2d(minibatch, filters=96,
                               kernel_size=[7, 7],
                               strides=[2, 2],
                               padding='valid',
                               data_format='channels_first',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.initializers.random_normal(
                                   stddev=0.01,
                                   seed=0),
                               bias_initializer=tf.initializers.constant(0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                   scale=10 ** (-5)),
                               reuse=tf.AUTO_REUSE,
                               name='conv1')

        out = tf.layers.batch_normalization(out, fused=True,
                                            renorm=True,
                                            name='batchnorm1')

        out = tf.layers.max_pooling2d(out, pool_size=[3, 3],
                                      strides=[3, 3],
                                      padding='valid',
                                      data_format='channels_first',
                                      name='pool1')

        out = tf.layers.conv2d(out, filters=256,
                               kernel_size=[7, 7],
                               strides=[1, 1],
                               padding='valid',
                               data_format='channels_first',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.initializers.random_normal(
                                   stddev=0.01,
                                   seed=0),
                               bias_initializer=tf.initializers.constant(0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                   scale=10 ** (-5)),
                               reuse=tf.AUTO_REUSE,
                               name='conv2')

        out = tf.layers.batch_normalization(out, fused=True,
                                            renorm=True,
                                            name='batchnorm2')

        out = tf.layers.max_pooling2d(out, pool_size=[2, 2],
                                      strides=[2, 2],
                                      padding='valid',
                                      data_format='channels_first',
                                      name='pool2')

        out = tf.layers.conv2d(out, filters=512,
                               kernel_size=[3, 3],
                               strides=[1, 1],
                               padding='same',
                               data_format='channels_first',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.initializers.random_normal(
                                   stddev=0.01,
                                   seed=0),
                               bias_initializer=tf.initializers.constant(0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                   scale=10 ** (-5)),
                               reuse=tf.AUTO_REUSE,
                               name='conv3')

        out = tf.layers.batch_normalization(out, fused=True,
                                            renorm=True,
                                            name='batchnorm3')

        out = tf.layers.conv2d(out, filters=512,
                               kernel_size=[3, 3],
                               strides=[1, 1],
                               padding='same',
                               data_format='channels_first',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.initializers.random_normal(
                                   stddev=0.01,
                                   seed=0),
                               bias_initializer=tf.initializers.constant(0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                   scale=10 ** (-5)),
                               reuse=tf.AUTO_REUSE,
                               name='conv4')

        out = tf.layers.batch_normalization(out, fused=True,
                                            renorm=True,
                                            name='batchnorm4')

        out = tf.layers.conv2d(out, filters=1024,
                               kernel_size=[3, 3],
                               strides=[1, 1],
                               padding='same',
                               data_format='channels_first',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.initializers.random_normal(
                                   stddev=0.01,
                                   seed=0),
                               bias_initializer=tf.initializers.constant(0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                   scale=10 ** (-5)),
                               reuse=tf.AUTO_REUSE,
                               name='conv5')

        out = tf.layers.batch_normalization(out, fused=True,
                                            renorm=True,
                                            name='batchnorm5')

        out = tf.layers.conv2d(out, filters=1024,
                               kernel_size=[3, 3],
                               strides=[1, 1],
                               padding='same',
                               data_format='channels_first',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.initializers.random_normal(
                                   stddev=0.01,
                                   seed=0),
                               bias_initializer=tf.initializers.constant(0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                   scale=10 ** (-5)),
                               reuse=tf.AUTO_REUSE,
                               name='conv6')

        out = tf.layers.batch_normalization(out, fused=True,
                                            renorm=True,
                                            name='batchnorm6')

        out = tf.layers.max_pooling2d(out, pool_size=[3, 3],
                                      strides=[3, 3],
                                      padding='valid',
                                      data_format='channels_first',
                                      name='pool3')

        out = tf.layers.flatten(out, name='flatten')

        out = tf.layers.dense(out, units=4096, activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.random_normal(
                                  stddev=0.01,
                                  seed=0),
                              bias_initializer=tf.initializers.constant(0),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                  scale=10 ** (-5)),
                              reuse=tf.AUTO_REUSE,
                              name='full1'
                              )

        out = tf.cond(self.mode,
                      lambda: tf.layers.dropout(
                          out, seed=0), lambda: out)

        out = tf.layers.dense(out, units=4096, activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.random_normal(
                                  stddev=0.01,
                                  seed=0),
                              bias_initializer=tf.initializers.constant(0),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                  scale=10 ** (-5)),
                              reuse=tf.AUTO_REUSE,
                              name='full2'
                              )
        out = tf.cond(self.mode,
                      lambda: tf.layers.dropout(
                          out, seed=0), lambda: out)
        logits = tf.layers.dense(out, units=self.numclasses,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.random_normal(
                                     stddev=0.01,
                                     seed=0),
                                 bias_initializer=tf.initializers.constant(0),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                     scale=10 ** (-5)),
                                 reuse=tf.AUTO_REUSE,
                                 name='output'
                                 )

        return logits
