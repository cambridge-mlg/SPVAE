import time
import tensorflow as tf
import glob
import pickle
import os
import numpy as np
from src.tfconvnet import ConvDenseArchitecture, BernoulliIWAE_conv, BinomialIWAE_conv, GaussianIWAE_conv
from src.model import DensityEstimatorModel
from scripts.experimentutils import get_datagraph
"""
Density estimation by CNN-VAE
"""

def _make_computationgraph(dataset="mnist", data_dir="Data", category=-1, dist_type="perturbed", nz=5,
                           take_frac=1.,n_dense=(16,16),stride=(2,2), padding=('valid', 'valid'),filters=(32,64),
                            kernel_size=(3, 3), batch_size=128, k_samples=5):
    # model will be built on top of the tf.data.Iterator
    dg = get_datagraph(dataset, data_dir, category, dist_type, batch_size, take_frac)

    #######################
    # Function Definition #
    #######################

    # Note: somnetimes, certain convolution parameters are not compatible with the image size
    archi = ConvDenseArchitecture(filters=filters, n_dense=n_dense, latent_nz=nz, stride=stride, padding=padding,
                                  kernel_size=kernel_size, )

    # SPN leaf depends on distribution choice
    if "binary" in dist_type:
        iwae = BernoulliIWAE_conv(architecture=archi, k_samples=k_samples)
    elif "perturb" in dist_type:
        iwae = GaussianIWAE_conv(architecture=archi, k_samples=k_samples)
    elif "discrete" in dist_type:
        iwae = BinomialIWAE_conv(architecture=archi, k_samples=k_samples)
    else:
        raise ValueError("Invalid distribution type")

    model_output = iwae(dg.next_batch)
    return dg, model_output

def get_results(dataset="mnist", data_dir="Data", category=-1, dist_type="perturbed", save_dir=str(time.time()), nz=5,
                take_frac=1.,n_dense=(16,16),stride=(2,2), padding=('valid', 'valid'),filters=(32,64),
                   kernel_size=(3, 3), batch_size=128, k_samples=5, learning_rate=0.001, validation_period=128,
                   strikes=5, sess=None):

    dg, model_output = _make_computationgraph(dataset, data_dir, category, dist_type, nz,
                    take_frac,n_dense,stride, padding,filters,
                   kernel_size, batch_size, k_samples)

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
                    "take_frac", "n_dense", "stride", "padding", "filters",
                    "kernel_size", "batch_size", "k_samples"]

            args = {a:attributes[a] for a in args if a in attributes}

            for k,v in modifications.items():
                args[k] = v

            dg, model_output = _make_computationgraph(**args)

            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

            # overwrite k_samples from 5 -> 5000
            for v in tf.global_variables():
                if "Variable" in v.name:
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