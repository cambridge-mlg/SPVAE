import tensorflow as tf
import os
import numpy as np

"""
Various ways to prepare mnist tf.data.Dataset
"""

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

    return tf.reshape(image_1d, [28*28])

def _read_batch_from_tfrecord(example_proto):
    """Custom decoder for the example protocol buffer"""

    print(example_proto)
    feature = {
        'image_raw': tf.FixedLenFeature(28*28, tf.string),
        # 'label': tf.FixedLenFeature([], tf.string)
    }

    features = tf.parse_example(example_proto, features=feature)

    # Since the arrays were stored as strings, they are now 1d
    # label_1d = tf.decode_raw(features['label'], tf.int64)
    print(features['image_raw'])
    image_1d = tf.decode_raw(features['image_raw'], tf.float32)
    # image_1d = features['image_raw']
    print(image_1d)
    return image_1d

def configure_tfrecordds(dataset, batch_size):
    dataset = dataset.map(_read_from_tfrecord).cache()
    # dataset = dataset.repeat().shuffle(10000)
    # dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000,))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def configure_npyds(dataset, batch_size):
    dataset = dataset.cache()
    # dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.repeat()
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000,))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def configure_constantds(dataset, batch_size):
    # dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000,))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def three_TFrecord_datasets(data_dir, dataset, category, distribution_type, train_batch_size, take_frac=1.):

    if category < 0:
        category = "combined"

    train_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, dataset,  distribution_type, str(category), "train.tfrecords")])
    validation_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, dataset,  distribution_type, str(category), "validation.tfrecords")])
    test_ds = tf.data.TFRecordDataset(
        [os.path.join(data_dir, dataset,  distribution_type, str(category), "test.tfrecords")])

    if category == "combined":
        # sizes = np.sum(np.load(os.path.join(data_dir, dataset, "dataset_size.npy")),axis=0)
        sizes = np.sum(np.genfromtxt(os.path.join(data_dir, dataset, "dataset_size.txt")), axis=0)
    else:
        # sizes = np.load(os.path.join(data_dir, dataset, "dataset_size.npy"))
        sizes = np.genfromtxt(os.path.join(data_dir, dataset, "dataset_size.txt"))[category, :]

    if take_frac<1.:
        take_count = int(take_frac * sizes[0])
        train_ds = train_ds.take(take_count)

    train_ds = configure_tfrecordds(train_ds, train_batch_size)
    validation_ds = configure_tfrecordds(validation_ds, sizes[1])
    test_ds = configure_tfrecordds(test_ds, sizes[2])

    print(train_batch_size, sizes[1],  sizes[2])
    return train_ds, validation_ds, test_ds

def three_npy_datasets(data_dir, dataset, category, distribution_type, train_batch_size, take_frac=1.):

    if category < 0:
        category = "combined"

    train_ph = tf.placeholder(name="train_data_ph", dtype=tf.float32)
    train_ds = tf.data.Dataset.from_tensor_slices(train_ph)
    train_npy = np.load(os.path.join(data_dir, dataset,  distribution_type, str(category), "train.npy"))

    validation_ph = tf.placeholder(name="val_data_ph", dtype=tf.float32)
    validation_ds = tf.data.Dataset.from_tensor_slices(validation_ph)
    validation_npy = np.load(os.path.join(data_dir, dataset,  distribution_type, str(category), "validation.npy"))

    test_ph = tf.placeholder(name="test_data_ph", dtype=tf.float32)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ph)
    test_npy = np.load(os.path.join(data_dir, dataset,  distribution_type, str(category), "test.npy"))

    if category == "combined":
        # sizes = np.sum(np.load(os.path.join(data_dir, dataset, "dataset_size.npy")),axis=0)
        sizes = np.sum(np.genfromtxt(os.path.join(data_dir, dataset, "dataset_size.txt")), axis=0)
    else:
        # sizes = np.load(os.path.join(data_dir, dataset, "dataset_size.npy"))
        sizes = np.genfromtxt(os.path.join(data_dir, dataset, "dataset_size.txt"))[category, :]


    if take_frac<1.:
        take_count = int(take_frac * sizes[0])
        train_ds = train_ds.take(take_count)

    train_ds = configure_npyds(train_ds, train_batch_size)
    validation_ds = configure_npyds(validation_ds, sizes[1])
    test_ds = configure_npyds(test_ds, sizes[2])

    print(train_batch_size, sizes[1],  sizes[2])
    return train_ds, validation_ds, test_ds, {train_ph: train_npy}, {validation_ph: validation_npy}, {test_ph: test_npy}

def three_constant_datasets(data_dir, dataset, category, distribution_type, train_batch_size, take_frac=1.):

    if category < 0:
        category = "combined"


    train_npy = np.load(os.path.join(data_dir, dataset,  distribution_type, str(category), "train.npy"))
    train_ds = tf.data.Dataset.from_tensor_slices(train_npy)

    validation_npy = np.load(os.path.join(data_dir, dataset, distribution_type, str(category), "validation.npy"))
    validation_ds = tf.data.Dataset.from_tensor_slices(validation_npy)

    test_npy = np.load(os.path.join(data_dir, dataset, distribution_type, str(category), "test.npy"))
    test_ds = tf.data.Dataset.from_tensor_slices(test_npy)


    if category == "combined":
        # sizes = np.sum(np.load(os.path.join(data_dir, dataset, "dataset_size.npy")),axis=0)
        sizes = np.sum(np.genfromtxt(os.path.join(data_dir, dataset, "dataset_size.txt")), axis=0)
    else:
        # sizes = np.load(os.path.join(data_dir, dataset, "dataset_size.npy"))
        sizes = np.genfromtxt(os.path.join(data_dir, dataset, "dataset_size.txt"))[category, :]


    if take_frac<1.:
        take_count = int(take_frac * sizes[0])
        train_ds = train_ds.take(take_count)

    train_ds = configure_constantds(train_ds, train_batch_size)
    validation_ds = configure_constantds(validation_ds, sizes[1]//10)
    test_ds = configure_constantds(test_ds, sizes[2]//10)

    print(train_batch_size, sizes[1],  sizes[2])
    return train_ds, validation_ds, test_ds
