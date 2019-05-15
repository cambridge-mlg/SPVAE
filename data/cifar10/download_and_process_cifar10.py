import numpy as np
from six.moves import urllib
import tarfile
import os
import tempfile
import shutil
import gzip

def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        print(filename + " already exists")
        return filepath
    if not os.path.exists(directory):
        os.makedirs(directory)
    url = 'https://www.cs.toronto.edu/~kriz/' + filename
    _, temp_filepath = tempfile.mkstemp()
    print('Downloading %s to %s' % (url, temp_filepath))
    urllib.request.urlretrieve(url, temp_filepath)
    print(os.path.getsize(temp_filepath))
    shutil.move(temp_filepath, filepath)
    return filepath

def un_gzip(in_file, out_file):
    with gzip.open(in_file, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_batch(filename):
    batch = np.load(filename, encoding='bytes')
    return batch[b"data"], batch[b"labels"], batch[b"filenames"]


if __name__=="__main__":
    # get root of project repo
    CIFAR10_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.dirname(CIFAR10_DIR)
    ROOT_DIR = os.path.dirname(DATA_DIR)
    import sys

    sys.path.append(ROOT_DIR)
    os.chdir(CIFAR10_DIR)

    import data.dataprocessing as dataproc

    data_filepath = download(".", "cifar-10-python.tar.gz")
    tar = tarfile.open(data_filepath)
    tar.extractall()
    tar.close()


    # Merge data batches

    train_datamatrix = []
    train_labelvector = []
    train_filenames = []
    for i in range(1, 6):
        data, labels, filenames = extract_batch("cifar-10-batches-py/data_batch_" + str(i))
        train_datamatrix.append(data)
        train_labelvector.append(labels)
        train_filenames += filenames

    train_datamatrix = np.concatenate(train_datamatrix, axis=0)
    train_labelvector = np.concatenate(train_labelvector, axis=0)
    test_datamatrix, test_labelvector, test_filenames = extract_batch("cifar-10-batches-py/test_batch")

    train_datamatrix = np.moveaxis(train_datamatrix.reshape((50000, 3, 32, 32)), 1, 3).reshape(50000, -1)
    test_datamatrix = np.moveaxis(test_datamatrix.reshape((10000, 3, 32, 32)), 1, 3).reshape(10000, -1)

    # make validation set
    if os.path.exists(os.path.join("indices", "combined", "train.indx")) and \
            os.path.exists(os.path.join("indices", "combined", "validation.indx")):
        print("Found existing indx files")
        # read indx file
        train_indices = np.genfromtxt(os.path.join("indices", "combined", "train.indx"), dtype=np.uint32)
        validation_indices = np.genfromtxt(os.path.join("indices", "combined", "validation.indx"), dtype=np.uint32)

    else:
        train_indices, validation_indices = dataproc.generate_train_val_idx(total = 50000, val=10000)
        # make dirs
        if not os.path.exists(os.path.join("indices", "combined")):
            print("Making dirs")
            os.makedirs(os.path.join("indices", "combined"))

        # test indices not needed
        np.savetxt(os.path.join("indices", "combined", "train.indx"), train_indices, fmt='%d', delimiter="\n")
        np.savetxt(os.path.join("indices", "combined", "validation.indx"), validation_indices, fmt='%d',
                   delimiter="\n")
        with open(os.path.join("indices", "combined", "train.names"), "w") as f:
            print("\n".join([train_filenames[idx].decode("utf-8")  for idx in train_indices]), file=f)
        with open(os.path.join("indices", "combined", "validation.names"), "w") as f:
            print("\n".join([train_filenames[idx].decode("utf-8") for idx in validation_indices]), file=f)

        dataproc.check_mutually_exclusive(train_indices, validation_indices)

    validation_datamatrix = train_datamatrix[validation_indices,:]
    validation_labelvector = train_labelvector[validation_indices]

    train_datamatrix = train_datamatrix[train_indices,:]
    train_labelvector = train_labelvector[train_indices]

    if not dataproc.isMinMaxScaled(train_datamatrix):
        print("MinMaxScaling train")
        train_datamatrix = dataproc.MinMaxScale(train_datamatrix)
    if not dataproc.isMinMaxScaled(validation_datamatrix):
        print("MinMaxScaling validation")
        validation_datamatrix = dataproc.MinMaxScale(validation_datamatrix)
    if not dataproc.isMinMaxScaled(test_datamatrix):
        print("MinMaxScaling test")
        test_datamatrix = dataproc.MinMaxScale(test_datamatrix)


    ##############
    # discretize #
    ##############
    # make dirs
    if not os.path.exists(os.path.join("discrete", "combined")):
        print("Making dirs")
        os.makedirs(os.path.join("discrete", "combined"))



    train_datamatrix = dataproc.discretize(train_datamatrix)
    validation_datamatrix = dataproc.discretize(validation_datamatrix)
    test_datamatrix = dataproc.discretize(test_datamatrix)

    print("Saving datamatrix to discrete/")
    np.save(os.path.join("discrete", "combined", "train"), train_datamatrix.astype(np.float32))
    np.save(os.path.join("discrete", "combined", "validation"), validation_datamatrix.astype(np.float32))
    np.save(os.path.join("discrete", "combined", "test"), test_datamatrix.astype(np.float32))

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

