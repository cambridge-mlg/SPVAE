import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# get root of experiment repo
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MAIN_DIR)
import sys
sys.path.append(ROOT_DIR)

import tensorflow as tf
from scripts import iwae
from scripts.experimentutils import save_attributes


def run(nz, nnodes, dataset="mnist", dist_type="perturbed"):
    """
    Constructs a tf.Graph and tf.Session to contain the instance of experiment
    """
    with tf.Graph().as_default():
        config = tf.ConfigProto(device_count={"CPU": 8}, intra_op_parallelism_threads=8,
                                inter_op_parallelism_threads=8)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        with tf.Session(config=config) as sess:
            arguments = {
                # experiment def / paths
                "dataset": dataset,
                "data_dir": os.path.join(ROOT_DIR, "data"),
                "category": -1,
                "dist_type": dist_type,
                "take_frac": 1,
                # model def
                "nz": nz,
                "batch_size": 128,
                "nnodes_recog": nnodes,
                "nnodes_gener": nnodes,
                "k_samples": 5,
                # learner def
                "learning_rate": 0.001,
                "validation_period": 128,
                "strikes": 10
            }

            current_time = int(time.time())

            save_dir = os.path.join(MAIN_DIR, "iwae", arguments["dataset"] + "_" + arguments["dist_type"],
                                    str(arguments["category"]),
                                    str(current_time))

            output = iwae.get_results(save_dir=save_dir, **arguments)

            save_attributes(save_dir, arguments, output)

    return output

if __name__ == "__main__":
    # Hyperparameters to tune
    try_nnodes = [(30, 30), (100, 100), (300, 300), (600, 600)]
    try_nz = [1, 2, 5, 25, 50, 100, 150, 200]

    for n_dense in try_nnodes:
        for nz in try_nz:
            try:
                print("========n_dense:{} nz: {}=========".format(n_dense, nz))
                run(nz, n_dense, "mnist", "perturbed")
            except Exception as e:
                print(e)
                continue


    for n_dense in try_nnodes:
        for nz in try_nz:
            try:
                print("========n_dense:{} nz: {}=========".format(n_dense, nz))
                run(nz, n_dense, "mnist", "discrete")
            except Exception as e:
                print(e)
                continue

    for n_dense in try_nnodes:
        for nz in try_nz:
            try:
                print("========n_dense:{} nz: {}=========".format(n_dense, nz))
                run(nz, n_dense, "svhn", "perturbed")
            except Exception as e:
                print(e)
                continue

    for n_dense in try_nnodes:
        for nz in try_nz:
            try:
                print("========n_dense:{} nz: {}=========".format(n_dense, nz))
                run(nz, n_dense, "svhn", "discrete")
            except Exception as e:
                print(e)
                continue

    for n_dense in try_nnodes:
        for nz in try_nz:
            try:
                print("========n_dense:{} nz: {}=========".format(n_dense, nz))
                run(nz, n_dense, "cifar10", "perturbed")
            except Exception as e:
                print(e)
                continue

    for n_dense in try_nnodes:
        for nz in try_nz:
            try:
                print("========n_dense:{} nz: {}=========".format(n_dense, nz))
                run(nz, n_dense, "cifar10", "discrete")
            except Exception as e:
                print(e)
                continue