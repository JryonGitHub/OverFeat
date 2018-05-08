import os
from colorama import Fore, Style
import glob
import tensorflow as tf
from data.datasets import traindb, valdb
from networks.overfeataccuratebase import OverFeatAccurateBase
import horovod.tensorflow as hvd

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

    net = OverFeatAccurateBase(inputdata[0], numclasses=1000)
    logits = net.logits
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='Loss')

    # Creating the optimizer (Only for Training)
    epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
    lr = tf.train.piecewise_constant(epoch, boundaries=[30, 50, 60, 70, 80],
                                     values=[0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625])
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.6)

    opt = hvd.DistributedOptimizer(opt)

    # Create the training op
    train_op = opt.minimize(loss)
    broadcast_op = hvd.broadcast_global_variables(root_rank=0)






