from trainbasenet import train_eval
from colorama import init
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

options_train = {
    'tfrecordpath': '/data/stars/share/STARSDATASETS/ILSVRC2017/tfrecords/train',
    'batchsize': 32,
    'checkpointpath': '/home/uujjwal/ujjwal-projects/OverFeat/checkpoints/overfeat-accurate.ckpt',
    'endepoch': 90,
    'logpath': '/home/uujjwal/ujjwal-projects/OverFeat/tboard/train',
    'display_step': 100

}

options_val = {
    'tfrecordpath': '/data/stars/share/STARSDATASETS/ILSVRC2017/tfrecords/val',
    'batchsize': 128,
    'logpath': '/home/uujjwal/ujjwal-projects/OverFeat/tboard/val',
    'display_step': 100
}

init()
train_eval(options_train, options_val)
