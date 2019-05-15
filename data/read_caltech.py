"""
Caltech101 TFRecords to Dataset objects
"""

import tensorflow as tf
import os
import numpy as np

"""
Need to know train and validation sizes
"""

# prepare datasets
def _read_from_tfrecord(example_proto):
    """Custom decoder for the example protocol buffer"""
    feature = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        # 'label': tf.FixedLenFeature([], tf.string)
    }

    features = tf.parse_example([example_proto], features=feature)

    # Since the arrays were stored as strings, they are now 1d
    # label_1d = tf.decode_raw(features['label'], tf.int64)
    image_1d = tf.decode_raw(features['image_raw'], tf.float32)

    return tf.reshape(image_1d, [25*25])
    # return image_1d

def configure_ds(dataset, batch_size):
    dataset = dataset.map(_read_from_tfrecord).cache()
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(100000,))
    return dataset.batch(batch_size).prefetch(1)

def three_ds(data_dir, category, distribution_type, train_batch_size):
    if category < 0:
        category = "combined"

    train_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, str(category), "train.tfrecords")])
    validation_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, str(category), "validation.tfrecords")])
    test_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, str(category), "test.tfrecords")])

    sizes = np.load(os.path.join(data_dir, "CALTECH", "indices", "dataset_size.npy"))[category, :]

    # Full batch learning if negative
    if train_batch_size<0:
        train_batch_size = sizes[0]

    train_ds = configure_ds(train_ds, train_batch_size)
    validation_ds = configure_ds(validation_ds, sizes[1])
    test_ds = configure_ds(test_ds, sizes[2])

    return train_ds, validation_ds, test_ds

# TODO: WARNING: untested function
def three_placeholder_ds(data_dir, distribution_type, train_batch_size):
    category_ph = tf.placeholder(dtype=tf.int32, name="cate")
    category_str = tf.py_func(lambda val: str(val), [category_ph], tf.string)
    train_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, category_str, "train.tfrecords")])
    validation_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, category_str, "validation.tfrecords")])
    test_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, category_str, "test.tfrecords")])

    # shape [category, (train size, validation size, test size)]
    sizes = np.load(os.path.join(data_dir, "CALTECH", "indices", "dataset_size.npy"))

    # Full batch learning
    if train_batch_size < 0:
        train_batch_size = tf.gather_nd(sizes, [category_ph, 0],  name="train_batch_size")
    val_batch_size = tf.gather_nd(sizes, [category_ph, 1],  name="val_batch_size")
    test_batch_size = tf.gather_nd(sizes, [category_ph, 2], name="test_batch_size")

    train_ds = configure_ds(train_ds, train_batch_size)
    validation_ds = configure_ds(validation_ds, val_batch_size)
    test_ds = configure_ds(test_ds, test_batch_size)

    return train_ds, validation_ds, test_ds, category_ph

# TODO: WARNING: untested function
def three_variable_ds(data_dir, distribution_type, train_batch_size):
    category = tf.get_variable(name="cate", shape = (1), dtype=tf.int32, initializer=0, trainable=False )

    def change_category(cate, sess):
        category.load(cate, sess)

    category_str = tf.py_func(lambda val: str(val), [category], tf.string)
    train_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, category_str, "train.tfrecords")])
    validation_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, category_str, "validation.tfrecords")])
    test_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "CALTECH", distribution_type, category_str, "test.tfrecords")])

    # shape [category, (train size, validation size, test size)]
    sizes = np.load(os.path.join(data_dir, "CALTECH", "indices", "dataset_size.npy"))

    # Full batch learning
    if train_batch_size < 0:
        train_batch_size = tf.gather_nd(sizes, [category, 0],  name="train_batch_size")
    val_batch_size = tf.gather_nd(sizes, [category, 1],  name="val_batch_size")
    test_batch_size = tf.gather_nd(sizes, [category, 2], name="test_batch_size")

    train_ds = configure_ds(train_ds, train_batch_size)
    validation_ds = configure_ds(validation_ds, val_batch_size)
    test_ds = configure_ds(test_ds, test_batch_size)

    return train_ds, validation_ds, test_ds, change_category

# TODO: WARNING: untested function
def all_three_ds(data_dir, distribution_type, train_batch_size):

    train_datasets = []
    validation_datasets = []
    test_datasets = []

    # shape [category, (train size, validation size, test size)]
    sizes = np.load(os.path.join(data_dir, "CALTECH", "indices", "dataset_size.npy"))

    for i in range(102):
        train_ds = tf.data.TFRecordDataset(
            [os.path.join(data_dir, "CALTECH", distribution_type, str(i), "train.tfrecords")])
        validation_ds = tf.data.TFRecordDataset(
            [os.path.join(data_dir, "CALTECH", distribution_type, str(i), "validation.tfrecords")])
        test_ds = tf.data.TFRecordDataset(
            [os.path.join(data_dir, "CALTECH", distribution_type, str(i), "test.tfrecords")])


        # Full batch learning
        if train_batch_size < 0:
            train_batch_size = tf.gather_nd(sizes, [i, 0],  name="train_batch_size")
        val_batch_size = tf.gather_nd(sizes, [i, 1],  name="val_batch_size")
        test_batch_size = tf.gather_nd(sizes, [i, 2], name="test_batch_size")

        train_ds = configure_ds(train_ds, train_batch_size)
        validation_ds = configure_ds(validation_ds, val_batch_size)
        test_ds = configure_ds(test_ds, test_batch_size)

        train_datasets.append(train_ds)
        validation_datasets.append(validation_ds)
        test_datasets.append(test_ds)

    return train_datasets, validation_datasets, test_datasets