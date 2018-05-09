import os
from colorama import Fore, Style
import glob
import tensorflow as tf
from data.datasets import traindb, valdb
from networks.overfeataccuratebase import OverFeatAccurateBase
import horovod.tensorflow as hvd
from utils.visualization import put_kernels_on_grid


def train_eval(option_train, option_val):
    hvd.init()
    numworkers = hvd.size()
    worker_index = hvd.local_rank()
    # Finding the training and validation files
    tfpath_train = option_train['tfrecordpath']
    trainfiles = glob.iglob(os.path.join(tfpath_train, '*.tfrecords'))
    trainfiles = list(trainfiles)
    numtrain = 1281167
    """
    for f in trainfiles:
        for _ in tf.python_io.tf_record_iterator(f):
            numtrain += 1
    """
    if worker_index == 0:
        tf.logging.info(
            Fore.GREEN + Style.BRIGHT + 'Found a total of {} training files with {} images.'.format(len(trainfiles),
                                                                                                    numtrain))

    tfpath_val = option_val['tfrecordpath']
    valfiles = glob.iglob(os.path.join(tfpath_val, '*.tfrecords'))
    valfiles = list(valfiles)
    numval = 50000
    """
    for f in valfiles:
        for _ in tf.python_io.tf_record_iterator(f):
            numval += 1
    """
    if worker_index == 0:
        tf.logging.info(
            Fore.GREEN + Style.BRIGHT + 'Found a total of {} validation files with {} images.'.format(len(valfiles),
                                                                                                      numval))

    # Creating training and validation datasets
    dbtrain = traindb(trainfiles, option_train['batchsize'], numworkers=numworkers, workerindex=worker_index)
    dbtrain_iter = dbtrain.make_initializable_iterator()
    tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Training dataset created')

    dbval = valdb(valfiles, option_val['batchsize'], numworkers=numworkers, workerindex=worker_index)
    dbval_iter = dbval.make_initializable_iterator()
    tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Validation dataset created')

    # Creating the network and loss
    netmode = tf.placeholder(dtype=tf.bool)
    traindata = dbtrain_iter.get_next()
    valdata = dbval_iter.get_next()

    inputdata = tf.cond(tf.equal(netmode, tf.constant(True)), lambda: traindata,
                        lambda: valdata)

    labels = inputdata[1]

    with tf.variable_scope('model'):
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

    # Get parameters to visualize
    # Only get it from worker 0
    if worker_index == 0:
        summaries_train = []
        kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'model/conv1/kernel')[0]
        kernel = put_kernels_on_grid(kernel)
        img_summary = tf.summary.image('model/conv1_filters', kernel)
        summaries_train.append(img_summary)
        for num in range(1, 7):
            kernel = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       'model/conv{}/kernel'.format(num))[0]
            hist_summary = tf.summary.histogram('model/conv{}_kernel'.format(num),
                                                kernel)
            summaries_train.append(hist_summary)
            bias = tf.get_collection(tf.GraphKeys.VARIABLES,
                                     'model/conv{}/bias'.format(num))[0]
            hist_summary = tf.summary.histogram('model/conv{}_bias'.format(num), bias)
            summaries_train.append(hist_summary)
        top1_reduced = hvd.allreduce(top1, average=True)
        top5_reduced = hvd.allreduce(top5, average=True)
        loss_reduced = hvd.allreduce(loss, average=True)
        loss_summary = tf.summary.scalar('Loss', loss_reduced)
        top1_summary = tf.summary.scalar('Top1_Error', top1_reduced)
        top5_summary = tf.summary.scalar('Top5_Error', top5_reduced)
        summaries_train += [loss_summary, top1_summary, top5_summary]
        summaries_train = tf.summary.merge(summaries_train)
        summaries_val = tf.summary.merge([loss_summary, top1_summary, top5_summary])

    latest_chkpt_train = tf.train.latest_checkpoint(option_train[
                                                        'checkpointpath'])

    tosave = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
    saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True, var_list=tosave)

    endepoch_train = option_train['endepoch']
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(worker_index)
    config.log_device_placement = False

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.local_variables_initializer()])

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(hvd.broadcast_global_variables(root_rank=0))
        try:
            if tf.train.checkpoint_exists(latest_chkpt_train):
                saver.restore(sess, latest_chkpt_train)
                modelname = sess.run(latest_chkpt_train)
                model_epoch = modelname[modelname.find('-') + 1:
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

        currepoch = sess.run(epoch)
        sess.run(dbtrain_iter.initializer)
        display_step = option_train['display_step']
        display_counter = 0
        iter_curr = currepoch * numtrain
        for epochiter in range(currepoch, endepoch_train):
            while True:
                try:
                    if display_counter % display_step == 0:
                        if worker_index == 0:
                            loss_value, top1_err, top5_err, eph, summaries, _, _, _ = sess.run(
                                [loss_reduced, top1_reduced, top5_reduced, epoch, summaries_train, train_op,
                                 top1_update,
                                 top5_update], feed_dict={net.mode: True, netmode: True})
                            tf.logging.info(
                                Fore.YELLOW + Style.BRIGHT + 'TRAIN : Epoch[{}], Iter[{}] - Loss={l:.3f], Top1 error={t1:.2f}, Top5 error={t5:.2f}'.format(
                                    eph, iter_curr, l=loss_value, t1=top1_err, t5=top5_err))
                            writer_train.add_summary(summaries_train, global_step=iter_curr)
                            writer_train.flush()
                            tf.logging.debug(
                                Fore.CYAN + Style.BRIGHT + 'TensorBoard file for training iteration {} has been flushed to disk.'.format(
                                    iter_curr))
                        else:
                            sess.run([train_op, top1_update, top5_update], feed_dict={net.mode: True, netmode: True})
                    else:
                        sess.run([train_op, top1_update, top5_update], feed_dict={net.mode: True, netmode: True})

                    iter_curr += 1
                    display_counter += 1
                except tf.errors.OutOfRangeError:
                    sess.run(dbval_iter.initializer)
                    display_counter = 0
                    display_step = option_val['display_step']
                    sess.run([reset_top1, reset_top5])
                    while True:
                        try:
                            if display_counter % display_step == 0:
                                if worker_index == 0:
                                    loss_value, top1_err, top5_err, eph, summaries, _, _ = sess.run(
                                        [loss_reduced, top1_reduced, top5_reduced, epoch, summaries_val,
                                         top1_update, top5_update], feed_dict={net.mode: False, netmode: False})
                                    tf.logging.info(Fore.YELLOW + Style.BRIGHT + 'VALIDATION : Epoch[{}], Iter[{}] - '
                                                                                 'Loss={l:.3f], Top1 error={t1:.2f}, '
                                                                                 'Top5 error={t5:.2f}'.format(
                                        eph, iter_curr, l=loss_value, t1=top1_err, t5=top5_err))
                                else:
                                    sess.run([top1_update, top5_update, summaries_val],
                                             feed_dict={net.mode: False, netmode: False})
                            else:
                                if worker_index == 0:
                                    _, _, summaries = sess.run([top1_update, top5_update, summaries_val],
                                                               feed_dict={net.mode: False, netmode: False})
                                else:
                                    sess.run([top1_update, top5_update, summaries_val],
                                             feed_dict={net.mode: False, netmode: False})

                            display_counter += 1
                        except tf.errors.OutOfRangeError:
                            writer_val.add_summary(summaries, global_step=iter_curr)
                            tf.logging.debug(
                                Fore.CYAN + Style.BRIGHT + 'TensorBoard file for validation for epoch {} has been flushed to disk.'.format(
                                    eph))
                            sess.run(dbtrain_iter.initializer)
                            display_counter = 0
                            display_step = option_train['display_step']
                            sess.run([reset_top5, reset_top1, epoch_change_op])
                            eph = sess.run(epoch)
                            if worker_index == 0:
                                saver.save(sess, option_train['savepath'], global_step=eph, write_meta_graph=False)
                                tf.logging.info(Fore.CYAN + Style.BRIGHT + 'Checkpoint for epoch {} saved.'.format(eph))
                            break
        if worker_index == 0:
            writer_train.close()
            writer_val.close()
    tf.logging.info(Fore.RED + Style.BRIGHT + 'Training finished.')
