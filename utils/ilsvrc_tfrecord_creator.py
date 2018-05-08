import argparse
import os
import tensorflow as tf
import numpy as np
import cv2
import multiprocessing as mp


def init_worker(shardscount, tfrfilename, basesavedir, fsplits):
    """
    An initializer for python multiprocessing.Pool.
    Declares a set of global variables which are used
    by all the processes during TFRecord creation.
    Hence, common information used by all the processes
    is transferred only once and this reduces
    inter-process communication and hence leads to speed-up.
    :param shardscount: Total number of shards to create.
    :param tfrfilename: Base name for the TFRecord file to be created. Example- if tfrfilename='mia' then
                        different shards will be named like mia-1024-of-1024.tfrecord. For details see
                        maketfr().
    :param basesavedir: Base path to the folder beneath which training and validation shards will be saved.
                        Example- if basesavedir='/home/ujjwal' then training shards are saved in '/home/ujjwal/train'
                        and validation shards are saved in '/home/ujjwal/val'. For details see maketfr()
    :param fsplits:     A list of tuples. Each tuple is like (<IMAGEFILE>,<IntegerLABEL>). The length of the list
                        is equal to the total number of file(s) (1281167 for ILSVRC training set and 50000 for ILSVRC
                        validation set).
    """

    global numshards, tfrname, basesave, splits

    numshards, tfrname, basesave, splits = shardscount, tfrfilename, basesavedir, fsplits


def _int64_feature(value):
    """
    Encodes an integer value as an int64List tf.train.Feature
    :param value: An integer.
    :return: A tf.train.Feature encoded as tf.train.Int64List
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    Encodes a ByteString value as a BytesList tf.train.Feature
    :param value: A Byte string.
    :return: A tf.train.Feature encoded as tf.train.BytesList
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(addr):
    """
    Performs a sequence of operations on an image.
    Specifically, it reads an image from disk and converts it to
    RGB channel order(as opposed to BGR order imposed by OpenCV).
    If the image is grayscale, it replicates the channel to make it
    3 channel.
    Finally, the image is resized with smaller dimension fixed to 256 pixels.
    The aspect ratio of the image is kept preserved.
    Bicubic interpolation is used while resizing to preserve the original
    quality of the image.
    :param addr: Full path to an image file on disk.
    :return: Processed image. See description above.
    """
    img = cv2.imread(addr)
    if len(img.shape) is not 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = img[..., ::-1]

    return img


def maketfr(shardnum):
    """
    Prepares a TFRecord file and saves it
    This function should only be used inside the framework of
    python's multiprocessing module. It makes use of several global
    variables which are passed as arguments to the initializer of
    multiprocessing.Pool constructor. For details of those arguments see init_worker()
    :param shardnum: An integer which identifies the sequence order of a shard.
    :return: None
    """
    filename = '{}-{:04d}-of-{}.tfrecords'.format(tfrname, shardnum, numshards)
    filename = os.path.join(basesave, filename)
    writer = tf.python_io.TFRecordWriter(os.path.join(basesave, filename))
    for i in splits[shardnum - 1]:
        img = i[0]
        label = i[1]
        img = load_image(img)
        height = img.shape[0]
        width = img.shape[1]
        img = cv2.imencode('.jpeg', img)[1].tostring()
        feature = {'label': _int64_feature(int(label)),
                   'image': _bytes_feature(img),
                   'width': _int64_feature(int(width)),
                   'height': _int64_feature(int(height)),
                   }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    print("Shard number {} of {} has been created with {} records.".format(shardnum, numshards,
                                                                           len(splits[shardnum - 1])))
    return None


def preparetfr(saveDir, devkitPath, trainlist, vallist, numshardsTrain, numshardsVal):
    """
    Prepares TFRecord for the ILSVRC Dataset (training and validation subsets for CLS-LOC competition).
    This function uses Python's multiprocessing module to parallelize the TFRecord preparation
    and hence speed up.
    :param saveDir: Base path to the folder beneath which training and validation shards will be saved.
                        Example- if basesavedir='/home/ujjwal' then training shards are saved in '/home/ujjwal/train'
                        and validation shards are saved in '/home/ujjwal/val'. For details see maketfr()
    :param ilsvrcPath: Base path to the ILSVRC Dataset. It must have the subfolder Data in which ILSVRC dataset
                       files are located.
    :param devkitPath: Base path to the ILSVRC devkit.
    :param trainlist: A text file which contains list of all the training image files of ILSVRC
    :param vallist: A text file which contains list of all the validation image files of ILSVRC
    :param numshardsTrain: Number of shards to prepare for training set.
    :param numshardsVal: Number of shards to prepare for validation set.
    :return: None
    """
    trainmapping = os.path.join(devkitPath, 'data', 'map_clsloc.txt')

    valmapping = os.path.join(devkitPath, 'data', 'ILSVRC2015_clsloc_validation_ground_truth.txt')

    train_tfrfilename = 'ilsvrc2017-cls-loc-train'

    val_tfrfilename = 'ilsvrc2017-cls-loc-val'

    if not os.path.exists(os.path.join(saveDir, 'train')):
        os.makedirs(os.path.join(saveDir, 'train'), exist_ok=True)

    if not os.path.exists(os.path.join(saveDir, 'val')):
        os.makedirs(os.path.join(saveDir, 'val'), exist_ok=True)

    print('Reading training label mapping.')

    trainmap = {}
    for line in open(trainmapping, 'r'):
        line = line.strip()
        wnid, label, _ = line.split(' ')
        wnid = wnid.strip()
        label = label.strip()
        trainmap[wnid] = int(label) - 1

    print("--------------------------------------------------------")
    print('Reading validatin label mapping.')

    valmap = []
    for line in open(valmapping, 'r'):
        line = line.strip()
        valmap.append(int(line) - 1)

    print("--------------------------------------------------------")

    print("Writing training data to {} shards.".format(numshardsTrain))

    with open(trainlist, 'r') as f:
        lines = f.read().splitlines()

    labels = []
    for line in lines:
        wnid = os.path.basename(os.path.dirname(line)).strip()
        labels.append(trainmap[wnid])

    traininfo = list(zip(lines, labels))

    trainsplits = np.array_split(traininfo, numshardsTrain)

    dirtrain = os.path.join(saveDir, 'train')

    seqtrain = []

    for i in range(numshardsTrain):
        seqtrain.append(i + 1)

    p = mp.Pool(processes=mp.cpu_count(), initializer=init_worker,
                initargs=(numshardsTrain, train_tfrfilename, dirtrain, trainsplits))

    for _ in p.imap_unordered(maketfr, seqtrain, chunksize=32):
        pass

    p.close()

    p.join()

    with open(vallist, 'r') as f:
        lines = f.read().splitlines()

    valinfo = list(zip(lines, valmap))

    valsplits = np.array_split(valinfo, numshardsVal)

    dirval = os.path.join(saveDir, 'val')

    seqtrain = []

    for i in range(numshardsVal):
        seqtrain.append(i + 1)

    p = mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(numshardsVal, val_tfrfilename, dirval, valsplits))

    for _ in p.imap_unordered(maketfr, seqtrain, chunksize=32):
        pass

    p.close()

    p.join()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This prepares sharded TFRecord files for ILSVRC CLS-LOC dataset.')

    parser.add_argument('-savedir', action="store", dest='saveDir', default="/data/stars/share/STARSDATASETS/ILSVRC2017/tfrecords")
    parser.add_argument('-devkitpath', action="store", dest='devkitPath',
                        default="/data/stars/share/STARSDATASETS/ILSVRC2017/devkit")
    parser.add_argument('-trainlist', action="store", dest='trainlist',
                        default="/data/stars/share/STARSDATASETS/ILSVRC2017/train_list.txt")
    parser.add_argument('-vallist', action="store", dest='vallist',
                        default="/data/stars/share/STARSDATASETS/ILSVRC2017/val_list.txt")
    parser.add_argument('-numshardstrain', action="store", dest='numshardsTrain', default=1024, type=int)
    parser.add_argument('-numshardsval', action="store", dest='numshardsVal', default=128, type=int)

    args = parser.parse_args()

    preparetfr(args.saveDir, args.devkitPath, args.trainlist, args.vallist, args.numshardsTrain,
               args.numshardsVal)
