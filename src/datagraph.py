import tensorflow as tf

"""
Motivation:
Feeding data into a model is one of the crucial steps in a machine learning pipeline
Tensorflow provides tf.data API as a standard yet flexible library for feeding data
tf.data consists of two component: Dataset and Iterator classes

For ML experimentation, these two objects are combined only in a common few patterns.
This file will implement these patterns of using the Dataset object and Iterator 
for the Training / Validation / Testing scenario.

Pre-requisites:
User knows how to set up a Dataset object.
(map, batch, repeat, shuffle) options etc
Otherwise, read: https://www.tensorflow.org/guide/datasets
"""

class SingleIteratorThreeDataset(object):
    def __init__(self, train_ds, val_ds, test_ds,
                 train_feeddict=None, val_feeddict=None, test_feeddict=None):
        self._iterator = tf.data.Iterator.from_structure(output_types=train_ds.output_types,
                                                         output_shapes=train_ds.output_shapes)

        self._next_batch = self._iterator.get_next(name="next_batch")

        # make three operators from Iterator
        self._train_init_op = self._iterator.make_initializer(train_ds)
        self._val_init_op = self._iterator.make_initializer(val_ds)
        self._test_init_op = self._iterator.make_initializer(test_ds)

        self.train_feeddict = train_feeddict
        self.val_feeddict = val_feeddict
        self.test_feeddict = test_feeddict

    def switch2trainds(self, sess):
        sess.run(self._train_init_op, feed_dict=self.train_feeddict)

    def switch2valds(self, sess):
        sess.run(self._val_init_op, feed_dict=self.val_feeddict)

    def switch2testds(self, sess):
        sess.run(self._test_init_op, feed_dict=self.test_feeddict)

    @property
    def next_batch(self):
        """build rest of computation graph on this placeholder"""
        return self._next_batch

class SingleIteratorSingleDataset(object):
    def __init__(self, train_filenames, val_filenames, test_filenames):
        """
        Reinitialisable iterator pattern.

        A training, validation, and testing datasets will have the same structure.
        So create an iterator and three initialization ops that link that iterator
        to one of the three datasets.

        :param train_filenames: list
        :param val_filenames: list
        :param test_filenames: list
        """

        # placeholder dataset
        # self._filenames = tf.placeholder(tf.string, shape=[None])
        self._filenames = tf.get_variable("handle", shape=[], dtype=tf.string, initializer=train_filenames[0], trainable=False)
        self._dataset = tf.data.TFRecordDataset(self._filenames)

        # setuo dataset options here
        # dataset = dataset.map(...)  # Parse the record into tensors.
        self._dataset = self._dataset.repeat()  # Repeat the input indefinitely.
        self._dataset = self._dataset.batch(32)

        # iterator from placeholder dataset
        self._iterator = self._dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next(name="next_batch")

        # filenames dont change during the experiments
        # so let's save it
        assert type(self.train_filename) == list
        assert type(self.val_filename) == list
        assert type(self.test_filename) == list
        self.train_filename = train_filenames
        self.val_filename = val_filenames
        self.test_filename = test_filenames

    def switch_newds(self, sess, filenames):
        """
        Initialize `iterator` with training data.
        :param sess:
        :param filenames: example: ["file1.tfrecord", "file2.tfrecord"]
        :return: None
        """
        # You can feed the initializer with the appropriate filenames for the current
        # phase of execution, e.g. training vs. validation.
        # sess.run(self._iterator.initializer, feed_dict={self._filenames: filenames})
        self._filenames.load(filenames, sess)

    def switch2trainds(self, sess):
        # sess.run(self._iterator.initializer, feed_dict={self._filenames: self.train_filename})
        self._filenames.load(self.train_filename, sess)
        sess.run(self._iterator.initializer)

    def switch2valds(self, sess):
        # sess.run(self._iterator.initializer, feed_dict={self._filenames: self.val_filename})
        self._filenames.load(self.val_filename, sess)
        sess.run(self._iterator.initializer)

    def switch2testds(self, sess):
        # sess.run(self._iterator.initializer, feed_dict={self._filenames: self.test_filename})
        self._filenames.load(self.test_filename, sess)
        sess.run(self._iterator.initializer)

    @property
    def next_batch(self):
        """build rest of computation graph on this placeholder"""
        return self._next_batch


class SingleIteratorAllThreeDataset(object):
    def __init__(self, train_datasets, validation_datasets, test_datasets):
        """
        This will create 300+ dataset objects. I hope the RAM can take it. Dont save the meta graph.
        This is better than accepting a new dataset at runtime because
        :param train_datasets:
        :param validation_datasets:
        :param test_datasets:
        """

        # check arguments
        assert len(train_datasets) == len(validation_datasets)
        assert len(train_datasets) == len(test_datasets)

        # initialize collection
        self._train_init_ops = []
        self._val_init_ops = []
        self._test_init_ops = []

        # the core iterator
        self._iterator = tf.data.Iterator.from_structure(output_types=train_datasets[0].output_types,
                                                         output_shapes=train_datasets[0].output_shapes)

        self._next_batch = self._iterator.get_next(name="next_batch")

        self._current_ds_indx = 0

        # make three operators from Iterator
        for train_ds, val_ds, test_ds in zip(train_datasets, validation_datasets, test_datasets):
            self._train_init_ops.append(self._iterator.make_initializer(train_ds))
            self._val_init_ops.append(self._iterator.make_initializer(val_ds))
            self._test_init_ops.append(self._iterator.make_initializer(test_ds))

    def switch2trainds(self, sess):
        sess.run(self._train_init_ops[self._current_ds_indx])

    def switch2valds(self, sess):
        sess.run(self._val_init_ops[self._current_ds_indx])

    def switch2testds(self, sess):
        sess.run(self._test_init_ops[self._current_ds_indx])

    def changeds(self, indx):
        assert type(indx) is int
        self._current_ds_indx = indx

    @property
    def next_batch(self):
        """build rest of computation graph on this placeholder"""
        return self._next_batch


class FourIteratorThreeDataset(object):
    def __init__(self, train_ds, val_ds, test_ds):
        """
        feedable iterator pattern
        :param train_ds:
        :param val_ds:
        :param test_ds:
        """

        self._train_iterator = train_ds.make_one_shot_iterator()

        self._train_iterator_handle = self._train_iterator.string_handle()
        self._val_iterator_handle = val_ds.make_one_shot_iterator().string_handle()
        self._test_iterator_handle = test_ds.make_one_shot_iterator().string_handle()


        # I chose to use a variable instead of an iterator so that I dont have to feed the iterator
        # at every step with the choice of dataset
        # I am assuming that this pattern will allow us to switch between Datasets even though we have not
        # finished iterating through an epoch

        # self.handle = tf.placeholder(tf.string, shape=[])
        self.handle = tf.get_variable("handle", shape=[], dtype=tf.string, initializer=self._train_iterator_handle, trainable=False)
        self._iterator = tf.data.Iterator.from_string_handle(
            self.handle, self._train_iterator.output_types)

        self._next_batch = self._iterator.get_next(name="next_batch")

    def switch2trainds(self, sess):
        self.handle.load(self._train_iterator_handle, sess)

    def switch2valds(self, sess):
        self.handle.load(self._val_iterator_handle, sess)

    def switch2testds(self, sess):
        self.handle.load(self._test_iterator_handle, sess)

    @property
    def next_batch(self):
        """build rest of computation graph on this placeholder"""
        return self._next_batch

class DataGraphSession(object):
    """
    For when I am lazy to implement shuffling and batching in pure python
    Exploit the functionality in tf.data and then import hptuning into python
    """
    def __init__(self, dataset, graph):
        """

        :param dataset:
        :param graph: the tf.Graph in which the dataset is defined

        graph = tf.Graph()
        with graph.as_default():
            # define dataset object here
            dataset = tf.data.Dataset.from_tensor_slices(datamatrix)
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_size,count))
            dataset = dataset.batch(32)

        """
        self._dataset = dataset
        self.graph = graph

        with graph.as_default():
            self._iterator = self._dataset.make_one_shot_iterator()
            self._next_batch = self._iterator.get_next(name="next_batch")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=self.graph, config=config)

    def next_batch(self):
        return self._session.run(self._next_batch)

