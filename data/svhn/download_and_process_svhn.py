import numpy as np
import scipy.io as spio
from six.moves import urllib
import os
import tempfile
import shutil

def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        print(filepath + " already exists")
        return filepath
    if not os.path.exists(directory):
        os.makedirs(directory)
    url = 'http://ufldl.stanford.edu/housenumbers/' + filename
    _, temp_filepath = tempfile.mkstemp()
    print('Downloading %s to %s' % (url, temp_filepath))
    urllib.request.urlretrieve(url, temp_filepath)
    print(os.path.getsize(temp_filepath))
    shutil.move(temp_filepath, filepath)
    return filepath


if __name__=="__main__":
    # get root of project repo
    SVHN_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.dirname(SVHN_DIR)
    ROOT_DIR = os.path.dirname(DATA_DIR)
    import sys

    sys.path.append(ROOT_DIR)
    os.chdir(SVHN_DIR)

    import data.dataprocessing as dataproc

    train_filepath = download(".", "train_32x32.mat")
    test_filepath = download(".", "test_32x32.mat")
    extra_filepath = download(".", "extra_32x32.mat")

    train_res = spio.loadmat(train_filepath)
    test_res = spio.loadmat(test_filepath)
    extra_res = spio.loadmat(extra_filepath)

    # check order of dimensions
    # shape == (height, width, colors, batch)
    print(train_res['X'].shape)
    print(train_res['y'].shape)
    print(test_res['X'].shape)
    print(test_res['y'].shape)
    print(extra_res['X'].shape)
    print(extra_res['y'].shape)

    # filter first 26032 examples of extra as validation set

    train_labelvector = train_res['y'].astype(np.uint8)
    test_labelvector = test_res['y'].astype(np.uint8)
    validation_labelvector = extra_res['y'][:26032, 0]

    # save labels as .npy file
    np.save(os.path.join("train_labels.npy"), train_labelvector)
    np.save(os.path.join("test_labels.npy"), test_labelvector)
    np.save(os.path.join("validation_labels.npy"), validation_labelvector)




    train_datatensor = np.moveaxis(train_res['X'], -1, 0)
    validation_datatensor = np.moveaxis(extra_res['X'], -1, 0)[:26032, :,:,:]
    test_datatensor = np.moveaxis(test_res['X'], -1, 0)

    train_datamatrix = dataproc.greyscalify(train_datatensor)
    validation_datamatrix = dataproc.greyscalify(validation_datatensor)
    test_datamatrix = dataproc.greyscalify(test_datatensor)

    if not dataproc.isMinMaxScaled(train_datamatrix):
        print("rescaled train datamatrix")
        train_datamatrix = dataproc.MinMaxScale(train_datamatrix)
    if not dataproc.isMinMaxScaled(test_datamatrix):
        print("rescaled test datamatrix")
        test_datamatrix = dataproc.MinMaxScale(test_datamatrix)
    if not dataproc.isMinMaxScaled(validation_datamatrix):
        print("rescaled validation datamatrix")
        validation_datamatrix = dataproc.MinMaxScale(validation_datamatrix)


    if not os.path.exists(os.path.join("discrete", "combined")):
        os.makedirs(os.path.join("discrete", "combined"))

    np.save(os.path.join("discrete", "combined", "train.npy"), train_datamatrix.astype(np.float32))
    np.save(os.path.join("discrete", "combined", "test.npy"), test_datamatrix.astype(np.float32))
    np.save(os.path.join("discrete", "combined", "validation.npy"), validation_datamatrix.astype(np.float32))

    # create dataset according to category for inspection
    for i in range(10):
        print("Saving datamatrix to discrete/" + str(i))

        train_datamatrix_i = train_datamatrix[train_labelvector == i, :]
        validation_datamatrix_i = validation_datamatrix[validation_labelvector == i, :]
        test_datamatrix_i = test_datamatrix[test_labelvector == i, :]

        print(train_datamatrix_i.shape, validation_datamatrix_i.shape, test_datamatrix_i.shape)

        if not os.path.exists(os.path.join("discrete", str(i))):
            os.makedirs(os.path.join("discrete", str(i)))
        np.save(os.path.join("discrete", str(i), "train"), train_datamatrix_i.astype(np.float32))
        np.save(os.path.join("discrete", str(i), "validation"), validation_datamatrix_i.astype(np.float32))
        np.save(os.path.join("discrete", str(i), "test"), test_datamatrix_i.astype(np.float32))

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