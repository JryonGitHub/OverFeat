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

    @property
    def logits(self):
        return self._logits

    def _bn(self, input, is_training, name):
        out = tf.layers.batch_normalization(input, fused=True, renorm=True, training=is_training,
                                      reuse=tf.AUTO_REUSE,
                                      name=name)
        return out


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
                                   scale=0.00001),
                               reuse=tf.AUTO_REUSE,
                               name='conv1')

        out = tf.layers.batch_normalization(out, axis=1, renorm=True, fused=True, name='batchnorm1', training=self.mode)

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
                                   scale=0.00001),
                               reuse=tf.AUTO_REUSE,
                               name='conv2')

        out = tf.layers.batch_normalization(out, axis=1, renorm=True, fused=True, name='batchnorm2', training=self.mode)



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
                                   scale=0.00001),
                               reuse=tf.AUTO_REUSE,
                               name='conv3')

        out = tf.layers.batch_normalization(out, axis=1, renorm=True, fused=True, name='batchnorm3', training=self.mode)


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
                                   scale=0.00001),
                               reuse=tf.AUTO_REUSE,
                               name='conv4')

        out = tf.layers.batch_normalization(out, axis=1, renorm=True, fused=True, name='batchnorm4', training=self.mode)



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
                                   scale=0.00001),
                               reuse=tf.AUTO_REUSE,
                               name='conv5')

        out = tf.layers.batch_normalization(out, axis=1, renorm=True, fused=True, name='batchnorm5', training=self.mode)



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
                                   scale=0.00001),
                               reuse=tf.AUTO_REUSE,
                               name='conv6')

        out = tf.layers.batch_normalization(out, axis=1, renorm=True, fused=True, name='batchnorm6', training=self.mode)



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
                                  scale=0.00001),
                              reuse=tf.AUTO_REUSE,
                              name='full1'
                              )

        out = tf.layers.batch_normalization(out, axis=-1, renorm=True, fused=True, name='batchnorm7', training=self.mode)



        out = tf.layers.dense(out, units=4096, activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.random_normal(
                                  stddev=0.01,
                                  seed=0),
                              bias_initializer=tf.initializers.constant(0),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                  scale=0.00001),
                              reuse=tf.AUTO_REUSE,
                              name='full2'
                              )

        out = tf.layers.batch_normalization(out, axis=-1, renorm=True, fused=True, name='batchnorm8', training=self.mode)



        logits = tf.layers.dense(out, units=self.numclasses,
                                 kernel_initializer=tf.initializers.random_normal(
                                     stddev=0.01,
                                     seed=0),
                                 bias_initializer=tf.initializers.constant(0),
                                 reuse=tf.AUTO_REUSE,
                                 name='output'
                                 )

        return logits
