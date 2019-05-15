"""
Common functions used in preprocessing data

Dataset is represented as a two dimensional matrix (numexamples, num_variables)

Oftened stored as a float32 because that is most convenient for GPUs
"""

import numpy as np
from PIL import Image
import cv2 as cv
import tensorflow as tf


#####################
# processing stages #
#####################

def isMinMaxScaled(datamatrix):
    per_image_max = np.max(datamatrix, axis=1)
    per_image_min = np.min(datamatrix, axis=1)
    return np.all(np.isclose(per_image_max, 1.)) and \
           np.all(np.isclose(per_image_min, 0))


def MinMaxScale(datamatrix):
    """
    change data type to float32
    linearly scale min pixel to 0. and max pixel to 1.
    """
    per_image_max = np.max(datamatrix, axis=1)
    per_image_min = np.min(datamatrix, axis=1)
    brigthnessrange = per_image_max - per_image_min
    datamatrix = ((datamatrix.T - per_image_min) / brigthnessrange).T

    return datamatrix


def isPerturbed(datamatrix):
    num_examples = len(datamatrix)
    for var_id in range(datamatrix.shape[1]):
        _, count = np.unique(datamatrix[:,var_id], return_counts=True)
        if len(count) < num_examples//3:
            return False
    return True


# the purpose of perturbing is to ensure that no pixel will have the exact same value
# across multiple examples
def perturb(datamatrix, N = 255):
    """requires the input to be minmaxscaled"""
    datamatrix = datamatrix + 1. / (N) * np.random.randn(*datamatrix.shape).astype(np.float32)

    # reflect
    datamatrix[datamatrix < 0] = -datamatrix[datamatrix < 0]
    datamatrix[datamatrix > 1] = 2. - datamatrix[datamatrix > 1]

    return datamatrix


def discretize(datamatrix, N=255):
    # discretize to (N+1) steps
    datamatrix = np.rint(datamatrix *N).astype(np.float64) / float(N)
    return datamatrix.astype(np.float32)

def isDiscretized(datamatrix, N=255):
    datamatrix = (datamatrix * N)
    residue = np.abs(datamatrix - np.rint(datamatrix))
    return np.max(residue) < 0.1



# convert to grayscale
def greyscalify(datatensor):
    """
    param datatensor: shape == (examples, height, width, colors)
        if datatensor is not in this format, use np.moveaxis
    """
    output_datamatrix = np.empty([len(datatensor), datatensor.shape[1] * datatensor.shape[2]])
    for i in range(len(datatensor)):
        imarr = datatensor[i, :, :, :]
        blackwhite = np.array(Image.fromarray(imarr).convert('L'))
        output_datamatrix[i, :] = blackwhite.reshape(-1)

    return output_datamatrix.astype(np.float32)


def isBinarized(datamatrix):
    mask1 = datamatrix > 0.1
    mask2 = datamatrix < 0.9
    return not np.any(np.logical_and(mask1, mask2))


def binarize_AG(datamatrix, reshape, blocksize, maxPixel=1):
    binary_datamatrix = np.empty(datamatrix.shape)
    for i in range(len(datamatrix)):
        img = datamatrix[i, :].reshape(reshape) * 255
        # function only accepts images with datatype uint8
        # so multiplying by 255 is necessary
        # maxPixel = 1 means that the output image will have max value of 1.
        IMG = cv.adaptiveThreshold(img.astype(np.uint8), maxPixel, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                   blocksize, 2)
        binary_datamatrix[i, :] = np.array(IMG).reshape(-1)
    assert (np.any(binary_datamatrix[i, :]) > 0)
    binary_datamatrix = binary_datamatrix.astype(np.float32)
    return binary_datamatrix

def binarize(datamatrix):
    binary_datamatrix = np.random.binomial(1,datamatrix)
    return binary_datamatrix.astype(np.float32)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(datamatrix, labelvector, filename):
    """Converts a dataset to tfrecords."""
    num_examples = len(datamatrix)

    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            image_raw = datamatrix[index, :].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature(int(labelvector[index])),
                        'image_raw': _bytes_feature(image_raw)
                    }))
            writer.write(example.SerializeToString())


def convert_to_i(datamatrix, filename):
    """Converts a dataset to tfrecords."""
    num_examples = len(datamatrix)

    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            image_raw = datamatrix[index, :].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': _bytes_feature(image_raw)
                    }))
            writer.write(example.SerializeToString())

def generate_train_val_idx(total=60000, val=10000):
    # generate random partition of indices
    all_indices = np.array(list(range(total)))
    np.random.shuffle(all_indices)
    validation_indices = all_indices[:val]
    train_indices = all_indices[val:]

    return train_indices, validation_indices

def check_mutually_exclusive(list1, list2):
    intersection = set(list1).intersection(set(list2))
    if intersection:
        print(intersection)
        raise ValueError