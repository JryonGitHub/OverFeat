import tensorflow as tf
import multiprocessing as mp
from data.preprocessing import preprocess_train, preprocess_val


def traindb(tfrecord_filelist, batchsize, numworkers, workerindex):
    tfrecord_filelist = tf.constant(tfrecord_filelist, dtype=tf.string)
    db = tf.data.TFRecordDataset(tfrecord_filelist, num_parallel_reads=500)
    db = db.shard(numworkers, workerindex)
    db = db.shuffle(buffer_size=5000)
    db = db.map(preprocess_train, num_parallel_calls=mp.cpu_count())
    db = db.prefetch(buffer_size=5000)
    db = db.batch(batch_size=batchsize)
    #db = db.apply(tf.contrib.data.prefetch_to_device('/gpu:{}'.format(workerindex)))
    return db


def valdb(tfrecord_filelist, batchsize, numworkers, workerindex):
    tfrecord_filelist = tf.constant(tfrecord_filelist, dtype=tf.string)
    db = tf.data.TFRecordDataset(tfrecord_filelist, num_parallel_reads=500)
    db = db.shard(numworkers, workerindex)
    db = db.shuffle(buffer_size=5000)
    db = db.map(preprocess_val, num_parallel_calls=mp.cpu_count())
    db = db.prefetch(buffer_size=5000)
    db = db.batch(batch_size=batchsize)
    #db = db.apply(tf.contrib.data.prefetch_to_device('/gpu:{}'.format(workerindex)))
    return db
