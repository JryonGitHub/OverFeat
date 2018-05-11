from trainbasenet import train_eval
from colorama import init
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

options_train = {
    'tfrecordpath': '/local/ujjwal/tfrecords/train',
    'batchsize': 256,
    'checkpointpath': '/home/uujjwal/ujjwal-projects/OverFeat/checkpoints',
    'endepoch': 90,
    'logpath': '/home/uujjwal/ujjwal-projects/OverFeat/tboard/train',
    'display_step': 100

}

options_val = {
    'tfrecordpath': '/local/ujjwal/tfrecords/val',
    'batchsize': 256,
    'logpath': '/home/uujjwal/ujjwal-projects/OverFeat/tboard/val',
    'display_step': 100
}

init()
train_eval(options_train, options_val)
