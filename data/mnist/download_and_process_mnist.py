#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# Modifications by Ping Liang Tan:
#     removed unzip functionality in download
#
import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf


def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
  """Validate that filename corresponds to images for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_images, unused
    rows = read32(f)
    cols = read32(f)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))
    if rows != 28 or cols != 28:
      raise ValueError(
          'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
          (f.name, rows, cols))


def check_labels_file_header(filename):
  """Validate that filename corresponds to labels for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_items, unused
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))


# this function follows a common pattern in software engineering
# of downloading to a temporary file and then moving the file.
# This is because downloads can fail mid way, thus corrupting any existing data / downstream processing.
# But moves are often atomic.
# tf.gfile.Open is used instead of a native python open because the destination may be
# google cloud storage. For this same reason, copy and delete of temp file is used
# instead of moving the temp file

def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    print(os.path.getsize(zipped_filepath))
    # with gzip.open(zipped_filepath, 'rb') as f_in, tf.gfile.Open(filepath + '.gz', 'wb') as f_out:
    #     shutil.copyfileobj(f_in, f_out)
    # os.remove(zipped_filepath)
    shutil.move(zipped_filepath, filepath)
    return filepath

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
    f: A file object that can be passed into a gzip reader.
    Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
    ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                        (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
    Returns:
    labels: a 1D uint8 numpy array.
    Raises:
    ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels

############################################################
# ^ Above is copied from tensorflow under Apache license ^ #
############################################################






if __name__ == "__main__":
    # get root of project repo
    MNIST_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.dirname(MNIST_DIR)
    ROOT_DIR = os.path.dirname(DATA_DIR)
    import sys

    sys.path.append(ROOT_DIR)
    os.chdir(MNIST_DIR)

    import data.dataprocessing as dataproc

    ############
    # download #
    ############

    # download mnist test set
    test_images_filepath = download(".", 't10k-images-idx3-ubyte.gz')
    test_labels_filepath = download(".", 't10k-labels-idx1-ubyte.gz')
    # download mnist train set
    train_images_filepath = download(".", 'train-images-idx3-ubyte.gz')
    train_labels_filepath = download(".", 'train-labels-idx1-ubyte.gz')

    with open(train_images_filepath, "rb") as f:
        train_images = extract_images(f)
    with open(test_images_filepath, "rb") as f:
        test_images = extract_images(f)
    with open(train_labels_filepath, "rb") as f:
        train_labels = extract_labels(f)
    with open(test_labels_filepath, "rb") as f:
        test_labels = extract_labels(f)

    # reshape
    train_images = train_images.reshape((60000,784))
    test_images = test_images.reshape((10000,784))

    # minmaxscale
    train_images = dataproc.MinMaxScale(train_images)
    test_images = dataproc.MinMaxScale(test_images)

    print(train_images.shape, test_images.shape, train_labels.shape, test_labels.shape)
    print(np.max(train_images), np.max(test_images))

    #########################################################
    # split orginal train set into train and validation set #
    #########################################################

    if os.path.exists(os.path.join("indices", "combined", "train.indx")) and \
            os.path.exists(os.path.join("indices", "combined", "validation.indx")):
        print("Found existing indx files")
        # read indx file
        train_indices = np.genfromtxt(os.path.join("indices", "combined", "train.indx"), dtype=np.uint32)
        validation_indices = np.genfromtxt(os.path.join("indices", "combined", "validation.indx"), dtype=np.uint32)

    else:
        train_indices, validation_indices = dataproc.generate_train_val_idx()
        # make dirs
        if not os.path.exists(os.path.join("indices", "combined")):
            print("Making dirs")
            os.makedirs(os.path.join("indices", "combined"))

        # test indices not needed
        np.savetxt(os.path.join("indices", "combined", "train.indx"), train_indices, fmt='%d', delimiter="\n")
        np.savetxt(os.path.join("indices", "combined", "validation.indx"), validation_indices, fmt='%d',
                   delimiter="\n")

    dataproc.check_mutually_exclusive(train_indices, validation_indices)

    validation_images = train_images[validation_indices,:]
    validation_labels = train_labels[validation_indices]

    train_images = train_images[train_indices,:]
    train_labels = train_labels[train_indices]


    print("Saving labelvector to discrete/")
    np.save("train_labels", train_labels.astype(np.uint8))
    np.save("validation_labels", validation_labels.astype(np.uint8))
    np.save("test_labels", test_labels.astype(np.uint8))

    ##############
    # discretize #
    ##############
    # make dirs
    if not os.path.exists(os.path.join("discrete", "combined")):
        print("Making dirs")
        os.makedirs(os.path.join("discrete", "combined"))

    train_datamatrix = dataproc.discretize(train_images)
    validation_datamatrix = dataproc.discretize(validation_images)
    test_datamatrix = dataproc.discretize(test_images)

    print("Saving datamatrix to discrete/")
    np.save(os.path.join("discrete", "combined", "train"), train_datamatrix.astype(np.float32))
    np.save(os.path.join("discrete", "combined", "validation"), validation_datamatrix.astype(np.float32))
    np.save(os.path.join("discrete", "combined", "test"), test_datamatrix.astype(np.float32))

    # create dataset according to category for inspection
    for i in range(10):
        print("Saving datamatrix to discrete/" + str(i))

        train_datamatrix_i = train_datamatrix[train_labels == i, :]
        validation_datamatrix_i = validation_datamatrix[validation_labels == i, :]
        test_datamatrix_i = test_datamatrix[test_labels == i, :]

        print(train_datamatrix_i.shape, validation_datamatrix_i.shape, test_datamatrix_i.shape)

        if not os.path.exists(os.path.join("discrete", str(i))):
            os.makedirs(os.path.join("discrete", str(i)))
        np.save(os.path.join("discrete", str(i), "train"), train_datamatrix_i.astype(np.float32))
        np.save(os.path.join("discrete", str(i), "validation"), validation_datamatrix_i.astype(np.float32))
        np.save(os.path.join("discrete", str(i), "test"), test_datamatrix_i.astype(np.float32))


    ###########
    # perturb #
    ###########
    # make dirs
    if not os.path.exists(os.path.join("perturbed", "combined")):
        print("Making dirs")
        os.makedirs(os.path.join("perturbed", "combined"))

    for subset, datamatrix in zip(["train", "validation", "test"],[train_datamatrix, validation_datamatrix, test_datamatrix]):
        if not dataproc.isMinMaxScaled(datamatrix):
            print("MinMaxScaling ... " + subset)
            raise Warning("Should already have been MinMaxScaled. What happened?")
        if not dataproc.isPerturbed(datamatrix):
            print("Perturbing ... " + subset)
            datamatrix = dataproc.perturb(datamatrix)
        if not dataproc.isMinMaxScaled(datamatrix):
            print("MinMaxScaling ... " + subset)
            datamatrix = dataproc.MinMaxScale(datamatrix)
        np.save(os.path.join("perturbed", "combined", subset), datamatrix.astype(np.float32))