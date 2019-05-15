from data import read_dataset
from src.datagraph import SingleIteratorThreeDataset
import numpy as np
import os
import pickle

"""
For the convenience of scripts scripts
"""

def get_datagraph(dataset, data_dir, category, dist_type, batch_size, take_frac = 1):
    """
    Unified interface to all the datasets prepared

    :param dataset: str
        name of the dataset
    :param data_dir: str
        where is the Data directory mounted?
    :param category: str
        desired subset of the dataset.
        Example: all the "0" in mnist or all the "1" in svhn
    :param dist_type: str:
        "perturbed", "discrete" or "binary"
    :param batch_size: int
        training batch size
        validation and test set are served as full batch
        TODO: serve validation and test set as minibatch
    :param take_frac: float
        fraction of training set to
    :return: tf.data.Iterator
    """
    print(dataset)
    # datasets
    if "mnist" == dataset:
        train_ds, validation_ds, test_ds = read_dataset.three_constant_datasets(data_dir, dataset, category, dist_type,
                                                                              batch_size, take_frac)

    elif "svhn" == dataset:
        train_ds, validation_ds, test_ds = read_dataset.three_constant_datasets(data_dir, dataset, category, dist_type,
                                                              batch_size, take_frac)

    elif "cifar10" == dataset:
        train_ds, validation_ds, test_ds = read_dataset.three_constant_datasets(data_dir, dataset, category, dist_type,
                                                               batch_size, take_frac)

    else:
        raise NotImplementedError
    # Iterator pattern
    dg = SingleIteratorThreeDataset(train_ds, validation_ds, test_ds)

    return dg

def save_attributes(save_dir, arguments, output):
    """

    :param save_dir: str
        where to save the attributes
    :param arguments:
        hyperparameters of the scripts
    :param output:
        results of the scripts
    :return:
    """
    attributes = {}

    # saving as single dimensional np array so that dtype can be easily inferred by pandas

    for k, v in arguments.items():
        if type(v) == int:
            attributes[k] = np.array(v, dtype=np.int64)
        elif type(v) == float:
            attributes[k] = np.array(v, dtype=np.float64)
        else:
            attributes[k] = v

    for k, v in output.items():
        if type(v) == int:
            attributes[k] = np.array(v, dtype=np.int64)
        elif type(v) == float:
            attributes[k] = np.array(v, dtype=np.float64)
        else:
            attributes[k] = v


    with open(os.path.join(save_dir, "attributes.pk"), "wb") as fhandle:
        pickle.dump(attributes, fhandle)