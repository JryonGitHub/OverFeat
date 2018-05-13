import os
import time
from colorama import Fore, Style
import glob
import tensorflow as tf
from data.datasets import traindb, valdb
from networks.overfeataccuratebase import OverFeatAccurateBase
# import horovod.tensorflow as hvd
from utils.visualization import put_kernels_on_grid


def activate_iterator(iterator):
    data = iterator.get_next()
    return data


def train_eval(option_train, option_val):
    # hvd.init()
    numworkers = 1
    worker_index = 0
    # Finding the training and validation files
    tfpath_train = option_train['tfrecordpath']
    trainfiles = glob.iglob(os.path.join(tfpath_train, '*.tfrecords'))
    trainfiles = list(trainfiles)
    trainfiles = trainfiles
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

    dbval = valdb(valfiles, option_val['batchsize'], numworkers=1, workerindex=0)
    dbval_iter = dbval.make_initializable_iterator()
    tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Validation dataset created')

    # Creating the network and loss
    netmode = tf.placeholder(dtype=tf.bool)
    traindata = dbtrain_iter.get_next()
    valdata = dbval_iter.get_next()

    inputdata = tf.cond(tf.equal(netmode, tf.constant(True)), lambda: activate_iterator(dbtrain_iter),
                        lambda: activate_iterator(dbval_iter))

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
    lr = tf.train.piecewise_constant(epoch, boundaries=[1, 2, 3, 4, 5, 30, 60],
                                     values=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.0001, 0.00001])
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    """
    filter_vars = ['model/batchnorm']
    trainvars = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        for filter_var in filter_vars:
            if filter_var not in var.name:
                trainvars.append(var)

    
    gradients = opt.compute_gradients(loss, var_list=trainvars)
    
    averaged_gradients = []
    for grad, var in gradients:
        if grad is not None:
            averaged_gradients.append((hvd.allreduce(grad), var))
        else:
            averaged_gradients.append((None, var))
    
    # opt = hvd.DistributedOptimizer(opt)
    """
    # Create the training op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss)
    # train_op = opt.apply_gradients(averaged_gradients)

    # Get parameters to visualize
    # Only get it from worker 0
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
    # top1 = hvd.allreduce(top1, average=True)
    # top1 = top1
    # top5 = hvd.allreduce(top5, average=True)
    # top5 = top5
    # loss = hvd.allreduce(loss, average=True)
    loss_summary = tf.summary.scalar('Loss', loss)
    top1_summary = tf.summary.scalar('Top1_Error', top1)
    top5_summary = tf.summary.scalar('Top5_Error', top5)
    summaries_train += [loss_summary, top1_summary, top5_summary]
    summaries_train = tf.summary.merge(summaries_train)
    summaries_val = tf.summary.merge([loss_summary, top1_summary, top5_summary])

    latest_chkpt_train = tf.train.latest_checkpoint(option_train[
                                                        'checkpointpath'])

    tosave = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
    saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True, var_list=tosave)

    endepoch_train = option_train['endepoch']
    config = tf.ConfigProto()
    config.log_device_placement = False
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.local_variables_initializer()])

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # sess.run(hvd.broadcast_global_variables(root_rank=0))
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
                                                                                worker_index))
                tf.logging.info(Fore.GREEN + Style.BRIGHT + 'Training will '
                                                            'begin from epoch {}'.format(model_epoch))
        except:
            if worker_index == 0:
                tf.logging.info(Fore.GREEN + Style.BRIGHT + 'No checkpoint '
                                                            'found. The model will be trained from scratch by all workers.')

        if worker_index == 0:
            writer_train = tf.summary.FileWriter(option_train['logpath'],
                                                 sess.graph)
            writer_val = tf.summary.FileWriter(option_val['logpath'])

        currepoch = sess.run(epoch)
        sess.run(dbtrain_iter.initializer)
        display_step = option_train['display_step']
        display_counter = 0
        iter_curr = currepoch * numtrain
        for epochiter in range(currepoch, endepoch_train):
            time_init = time.time()
            while True:
                try:
                    if display_counter % display_step == 0:
                        _, loss_value, top1_err, top5_err, eph, summaries, _, _, _ = sess.run(
                            [update_ops, loss, top1, top5, epoch, summaries_train, train_op,
                             top1_update,
                             top5_update], feed_dict={net.mode: True, netmode: True})
                        tf.logging.info(
                            Fore.YELLOW + Style.BRIGHT + 'TRAIN : Epoch[{}], Iter[{}] Time for {} iterations[{ttaken:.3f}sec]- Loss={lval:.3f}, Top1 error={t1:.2f}, Top5 error={t5:.2f}'.format(
                                eph, iter_curr, display_step, ttaken=time.time() - time_init, lval=loss_value,
                                t1=top1_err, t5=top5_err))
                        writer_train.add_summary(summaries, global_step=iter_curr)
                        writer_train.flush()
                        tf.logging.debug(
                            Fore.CYAN + Style.BRIGHT + 'TensorBoard file for training iteration {} has been flushed to disk.'.format(
                                iter_curr))
                        time_init = time.time()

                    else:
                        sess.run([update_ops, train_op, top1_update, top5_update],
                                 feed_dict={net.mode: True, netmode: True})

                    iter_curr += 1
                    display_counter += 1
                except tf.errors.OutOfRangeError:
                    sess.run(dbval_iter.initializer)
                    display_counter = 0
                    display_step = option_val['display_step']
                    sess.run([reset_top1, reset_top5])
                    time_init = time.time()
                    while True:
                        try:
                            if display_counter % display_step == 0:
                                loss_value, top1_err, top5_err, eph, summaries, _, _ = sess.run(
                                    [loss, top1, top5, epoch, summaries_val,
                                     top1_update, top5_update], feed_dict={net.mode: False, netmode: False})
                                tf.logging.info(
                                    Fore.YELLOW + Style.BRIGHT + 'VALIDATION : Epoch[{}], Iter[{}] Time for {} iterations[{ttaken:.3f}sec] - '
                                                                 'Loss={lval:.3f}, Top1 error={t1:.2f}, '
                                                                 'Top5 error={t5:.2f}'.format(
                                        eph, iter_curr, display_step, ttaken=time.time() - time_init, lval=loss_value,
                                        t1=top1_err, t5=top5_err))
                                time_init = time.time()
                            else:
                                _, _, summaries = sess.run([top1_update, top5_update, summaries_val],
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
                            saver.save(sess, os.path.join(option_train['checkpointpath'], 'overfeat_accurate.ckpt'),
                                       global_step=eph, write_meta_graph=False)
                            tf.logging.info(Fore.CYAN + Style.BRIGHT + 'Checkpoint for epoch {} saved.'.format(eph))
                            break
        if worker_index == 0:
            writer_train.close()
            writer_val.close()
    tf.logging.info(Fore.RED + Style.BRIGHT + 'Training finished.')
