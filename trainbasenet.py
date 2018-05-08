import os
from colorama import Fore, Style
import glob
import tensorflow as tf
from data.datasets import traindb, valdb
from networks.overfeataccuratebase import OverFeatAccurateBase
import horovod.tensorflow as hvd
from utils.visualization import put_kernels_on_grid

def train_eval(option_train, option_val):

    # Finding the training and validation files
    tfpath_train = option_train['tfrecordpath']
    trainfiles = glob.iglob(os.path.join(tfpath_train, '*.tfrecords'))
    trainfiles = list(trainfiles)
    tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Found a total of {} training files.'.format(len(trainfiles)))

    tfpath_val = option_val['tfrecordpath']
    valfiles = glob.iglob(os.path.join(tfpath_val, '*.tfrecords'))
    valfiles = list(valfiles)
    tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Found a total of {} validation files.'.format(len(valfiles)))

    # Creating training and validation datasets
    dbtrain = traindb(trainfiles, option_train['batchsize'], numworkers=hvd.size(), workerindex=hvd.local_rank())
    dbtrain_iter = dbtrain.make_initializable_iterator()
    tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Training dataset created')

    dbval = valdb(valfiles, option_val['batchsize'], numworkers=hvd.size(), workerindex=hvd.local_rank())
    dbval_iter = dbval.make_initializable_iterator()
    tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Validation dataset created')

    # Creating the network and loss
    netmode = tf.placeholder(dtype=tf.bool)
    traindata = dbtrain_iter.get_next()
    valdata = dbval_iter.get_next()

    inputdata = tf.cond(tf.equal(netmode, tf.constant(True)), lambda: traindata,
                        lambda : valdata)

    labels = tf.one_hot(inputdata[1], 1000, name='one_hot_labels')

    with tf.name_scope('model'):
        net = OverFeatAccurateBase(inputdata[0], numclasses=1000)
    logits = net.logits
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='Loss')
    loss = tf.reduce_mean(loss, keepdims=None)
    predictions = tf.nn.softmax(logits)

    # Creating the metrics for evaluation
    with tf.variable_scope('top1') as scope:
        top1, top1_update = tf.metrics.mean(tf.nn.in_top_k(
            predictions=predictions, targets=labels, k=1),
                                                name='top1')
        top1 = tf.subtract(tf.constant(1.0, dtype=tf.float32), top1)
        vars = tf.contrib.framework.get_variables(scope,
                                                  collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_top1 = tf.variables_initializer(vars)

    with tf.variable_scope('top5') as scope:
        top5, top5_update = tf.metrics.mean(tf.nn.in_top_k(
            predictions=predictions, targets=labels, k=5),
                                                name='top5')
        top5 = tf.subtract(tf.constant(1.0, dtype=tf.float32), top5)
        vars = tf.contrib.framework.get_variables(scope,
                                                  collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_top5 = tf.variables_initializer(vars)

    # Creating the optimizer (Only for Training)
    epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
    epoch_change_op = tf.assign_add(epoch, 1)
    epoch_holder = tf.placeholder(dtype=tf.int32, shape=())
    epoch_assign_op = tf.assign(epoch, epoch_holder)
    lr = tf.train.piecewise_constant(epoch, boundaries=[30, 50, 60, 70, 80],
                                     values=[0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625])
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.6)

    opt = hvd.DistributedOptimizer(opt)

    # Create the training op
    train_op = opt.minimize(loss)
    broadcast_op = hvd.broadcast_global_variables(root_rank=0)

    # Get parameters to visualize
    # Only get it from worker 0
    if hvd.local_rank() == 0:
        summaries_train = []
        kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
        kernel = put_kernels_on_grid(kernel)
        img_summary = tf.summary.image('conv1_filters', kernel)
        summaries_train.append(img_summary)
        for num in range(1, 7):
            kernel = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       'conv{}/kernel'.format(num))[0]
            hist_summary = tf.summary.histogram('conv{}_kernel'.format(num),
                                                kernel)
            summaries_train.append(hist_summary)
            bias = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       'conv{}/bias'.format(num))[0]
            hist_summary = tf.summary.histogram('conv{}_bias'.format(num), bias)
            summaries_train.append(hist_summary)
        loss_reduced = hvd.allreduce(loss, average=True)
        top1_reduced = hvd.allreduce(top1, average=True)
        top5_reduced = hvd.allreduce(top5, average=True)
        loss_summary = tf.summary.scalar('Loss', loss_reduced)
        top1_summary = tf.summary.scalar('Top1_Error', top1_reduced)
        top5_summary = tf.summary.scalar('Top5_Error', top5_reduced)
        summaries_train+=[loss_summary, top1_summary, top5_summary]
        summaries_train = tf.summary.merge(summaries_train)
        summaries_val = tf.summary.merge([loss_summary, top1_summary, top5_summary])

    latest_chkpt_train = tf.train.latest_checkpoint(option_train[
                                                        'checkpointpath'])
    saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True)

    tosave = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.local_variables_initializer()])

    with tf.Session() as sess:
        sess.run(broadcast_op)
        sess.run(init_op)
        try:
            if tf.train.checkpoint_exists(latest_chkpt_train):
                saver.restore(sess, latest_chkpt_train)
                modelname = sess.run(latest_chkpt_train)
                model_epoch = modelname[modelname.find('-')+1 :
                                        modelname.find('.')] + 1
                sess.run(epoch_assign_op, feed_dict={epoch_holder:
                                                         model_epoch})
                tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Model restored ' \
                                                          'from {} by '
                                             'worker {}.'.format(latest_chkpt_train,
                                                          hvd.rank()))
                tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Training will '
                                                            'begin from epoch {}'.format(model_epoch))
        except:
            if hvd.rank() == 0:
                tf.logging.info(Fore.GREEN + Style.BRIGHT + 'No checkpoint '
                                                            'found. The model will be trained from scratch by all workers.')

        if hvd.rank() == 0:
            writer_train = tf.summary.FileWriter(option_train['logpath'],
                                                 sess.graph)
            writer_val = tf.summary.FileWriter(option_val['logpath'])

        sess.run(dbtrain_iter.initializer)













