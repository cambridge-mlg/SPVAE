import time
import os
import glob
import pickle
import tensorflow as tf
import numpy as np
from src.tflayer_spn import SPNTFLayer
from src.tflayer_leaf import IWAETFLayer_Factory_globalinference
from src.tfdensitynet import BernoulliIWAE, GaussianIWAE, BinomialIWAE, DenseNetArchitecture
from src.model import DensityEstimatorModel
from src.scope import PoonRegionGraph
from scripts.experimentutils import get_datagraph

"""
Density estimation by SP-IWAE 
"""

def _make_computationgraph(dataset, data_dir, category, dist_type,
                   take_frac, batch_size, nnodes_recog, nnodes_gener,
                   nz, k_samples, sumK, leafK):

    # model will be built on top of the tf.data.Iterator
    dg = get_datagraph(dataset, data_dir, category, dist_type, batch_size, take_frac)

    #######################
    # Function Definition #
    #######################

    recog_archi = DenseNetArchitecture(n_hidden=nnodes_recog, n_output=nz)

    if dataset == "caltech":
        poon = PoonRegionGraph(25, 25, [5])
        gener_archi = DenseNetArchitecture(n_hidden=nnodes_gener, n_output=25)


    elif dataset == "mnist":
        poon = PoonRegionGraph(28, 28, [7])
        gener_archi = DenseNetArchitecture(n_hidden=nnodes_gener, n_output=49)

    elif dataset == "mnist2":
        poon = PoonRegionGraph(28, 28, [14])
        gener_archi = DenseNetArchitecture(n_hidden=nnodes_gener, n_output=196)

    elif dataset == "mnist3":
        poon = PoonRegionGraph(28, 28, [4])
        gener_archi = DenseNetArchitecture(n_hidden=nnodes_gener, n_output=16)

    elif dataset == "svhn":
        poon = PoonRegionGraph(32, 32, [8])
        gener_archi = DenseNetArchitecture(n_hidden=nnodes_gener, n_output=64)

    elif dataset == "svhn2":
        poon = PoonRegionGraph(32, 32, [16])
        gener_archi = DenseNetArchitecture(n_hidden=nnodes_gener, n_output=256)

    elif dataset == "cifar10":
        poon = PoonRegionGraph(32, 32, [8])
        gener_archi = DenseNetArchitecture(n_hidden=nnodes_gener, n_output=64)

    else:
        raise NotImplementedError


    # SPN leaf depends on distribution choice
    if "binary" in dist_type:
        leaflayer = IWAETFLayer_Factory_globalinference(dg.next_batch, recog_archi, gener_archi, BernoulliIWAE, k_samples=k_samples)
    elif "perturb" in dist_type:
        leaflayer = IWAETFLayer_Factory_globalinference(dg.next_batch, recog_archi, gener_archi, GaussianIWAE, k_samples=k_samples)
    elif "discrete" in dist_type:
        leaflayer = IWAETFLayer_Factory_globalinference(dg.next_batch, recog_archi, gener_archi, BinomialIWAE, k_samples=k_samples)
    else:
        raise ValueError("Invalid distribution type")

    model_output = SPNTFLayer(K=1, scope_id=poon.root.id, region_graph=poon, leaflayer=leaflayer,
                              sumK=sumK, leafK=leafK)(dg.next_batch)

    return dg, model_output


def get_results(dataset="mnist", data_dir="Data", save_dir=str(time.time()), category=-1, dist_type="perturbed",
                   take_frac = 1., batch_size=128, nnodes_recog=(25,25),nnodes_gener=(25,25),
                   nz=1, k_samples=5, sumK=2, leafK=2, learning_rate=0.01, validation_period=128, strikes=5,
                   sess=None):
    dg, model_output = _make_computationgraph(dataset, data_dir, category, dist_type,
                   take_frac, batch_size, nnodes_recog, nnodes_gener,
                   nz, k_samples, sumK, leafK)

    ####################
    # Experiment Setup #
    ####################
    model = DensityEstimatorModel(dg.next_batch, model_output)
    model.compile()
    model.init_session(session=sess)
    model.init_trainingutils(save_dir=save_dir)
    output = model.fit_earlystopping(dg,
                                     learning_rate=learning_rate,
                                     validation_period=validation_period,
                                     maxStrikes=strikes)

    return output


def get_evaluation(save_dir, Nsamples=5000, modifications=None, batch_size=1):
    """

    :param save_dir:
    :param Nsamples:
    :param modifications: dict
        if there are errors or missing keys in the attributes.pk file,
        use this argument to overwrite it
    :param batch_size:
    :return:
    """

    with tf.Graph().as_default():
        config = tf.ConfigProto(device_count={"CPU": 8}, intra_op_parallelism_threads=8,
                                inter_op_parallelism_threads=8)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        with tf.Session(config=config) as sess:
            with open(os.path.join(save_dir, "attributes.pk"), "rb") as f:
                attributes = pickle.load(f)

            checkpoint_path = sorted(glob.glob(os.path.join(save_dir, "saved_models", "model.ckpt*.data-*")))[-1].split(".data-")[0]

            args = ["dataset", "data_dir", "category", "dist_type", "nz",
                   "take_frac", "batch_size", "nnodes_recog", "nnodes_gener", "k_samples", "sumK", "leafK"]

            args = {a:attributes[a] for a in args if a in attributes}

            for k,v in modifications.items():
                args[k] = v

            dg, model_output = _make_computationgraph(**args)

            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

            # overwrite k_samples from 5 -> 5000
            for v in tf.global_variables():
                if "Variable" in v.name:
                    assert "Leaf" in v.name
                    v.load(Nsamples, sess)

            dg.switch2testds(sess)


            outputs = []
            for test_batch in range(10):
                test_ds = sess.run(dg.next_batch)
                for i in range(len(test_ds) // batch_size):
                    print(str(i) + " "*10, end="\r")
                    feed_dict = {dg.next_batch: test_ds[i * batch_size:(i + 1) * batch_size, :]}
                    outputs.append(sess.run(model_output, feed_dict=feed_dict))

            outputs = np.concatenate(outputs)
            return np.mean(outputs)
