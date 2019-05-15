import tensorflow as tf
import os
import numpy as np
"""
Various ways to prepare svhn tf.data.Dataset
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

    return tf.reshape(image_1d, [32*32])

def configure_ds(dataset, batch_size):
    dataset = dataset.map(_read_from_tfrecord).cache()
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(100000,))
    return dataset.batch(batch_size).prefetch(1)

def configure_npyds(dataset, batch_size):
    dataset = dataset.cache()
    # dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.repeat()
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000,))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def three_ds(data_dir, category, distribution_type, train_batch_size, take_frac=1.):

    if category < 0:
        category = "combined"

    train_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "svhn", distribution_type, "{}".format(category), "train.tfrecords")])
    validation_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "svhn", distribution_type, "{}".format(category), "validation.tfrecords")])
    test_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, "svhn", distribution_type, "{}".format(category), "test.tfrecords")])

    # shape [category, (train size, validation size, test size)]
    if category == "combined":
        sizes = np.sum(np.load(os.path.join(data_dir, "svhn",  "dataset_size.npy")),axis=0)
    else:
        sizes = np.load(os.path.join(data_dir, "svhn", "dataset_size.npy"))

    # Full batch learning
    if train_batch_size<0:
        train_batch_size = sizes[0]

    if take_frac<1.:
        take_count = int(take_frac * sizes[0])
        train_ds = train_ds.take(take_count)

    train_ds = configure_ds(train_ds, train_batch_size)
    validation_ds = configure_ds(validation_ds, sizes[1])
    test_ds = configure_ds(test_ds, sizes[2])

    return train_ds, validation_ds, test_ds

def three_npy_datasets(data_dir, category, distribution_type, train_batch_size, take_frac=1.):

    if category < 0:
        category = "combined"

    train_ph = tf.placeholder(name="train_data_ph", dtype=tf.float32)
    train_ds = tf.data.Dataset.from_tensor_slices(train_ph)
    train_npy = np.load(os.path.join(data_dir, "svhn", distribution_type, str(category), "train.npy"))

    validation_ph = tf.placeholder(name="val_data_ph", dtype=tf.float32)
    validation_ds = tf.data.Dataset.from_tensor_slices(validation_ph)
    validation_npy = np.load(os.path.join(data_dir, "svhn", distribution_type, str(category), "validation.npy"))

    test_ph = tf.placeholder(name="test_data_ph", dtype=tf.float32)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ph)
    test_npy = np.load(os.path.join(data_dir, "svhn", distribution_type, str(category), "test.npy"))

    if category == "combined":
        sizes = np.sum(np.load(os.path.join(data_dir, "svhn", "dataset_size.npy")),axis=0)
    else:
        sizes = np.load(os.path.join(data_dir, "svhn", "dataset_size.npy"))[category, :]


    if take_frac<1.:
        take_count = int(take_frac * sizes[0])
        train_ds = train_ds.take(take_count)

    train_ds = configure_npyds(train_ds, train_batch_size)
    validation_ds = configure_npyds(validation_ds, sizes[1])
    test_ds = configure_npyds(test_ds, sizes[2])

    print(train_batch_size, sizes[1],  sizes[2])
    return train_ds, validation_ds, test_ds, {train_ph: train_npy}, {validation_ph: validation_npy}, {test_ph: test_npy}