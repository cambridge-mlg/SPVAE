import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def KL_diagonalgaussians(mu1, inv_sigma_sq1, mu2, inv_sigma_sq2):
    term1 = 0.5 * (-np.log(inv_sigma_sq2) + np.log(inv_sigma_sq1))
    term2 = 0.5 * inv_sigma_sq2 / inv_sigma_sq1
    term3 = 0.5 * np.square(mu1-mu2) * inv_sigma_sq2
    term4 = -0.5

    return term1 + term2 + term3 + term4



class DensityEstimatorModel(object):
    """
    Follows keras' model API.
    Need to reimplement keras api because
    I don't know how to use keras's api for density estimation.

    """

    def __init__(self, input, output):
        """

        """
        self.input = input
        self.output = output
        self.debug = True

    def compile(self, optimizer=tf.train.AdamOptimizer):
        """
        After a model has been trained, the operations defined here would not be needed
        So this Model is only useful for training/validation/testing.
        When a model is to be deployed, please rebuild another graph without these bells and whistles

        :param optimizer:
        :return:
        """
        print("Compilation Started!")

        starttime = time.time()

        # key metrics
        self.mean_loglikelihood = tf.reduce_mean(self.output, name="loglikelihood")
        self.reg_loss = tf.losses.get_regularization_loss(name="reg_loss")
        self.cost = tf.add(-self.mean_loglikelihood, self.reg_loss, name="cost")

        # summary stuff
        tf.summary.scalar(name="loglikelihood", tensor=self.mean_loglikelihood)  # name might cause an issue
        tf.summary.scalar(name="cost", tensor=self.cost)  # name might cause an issue
        self.summary_op = tf.summary.merge_all()  # name="summary_op"

        # training stuff
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="learning_rate_ph")
        self.train_op = optimizer(learning_rate=self.learning_rate_ph, name="train_op").minimize(self.cost)

        # assign op to initialize variables with some (perhaps random) numbers
        self.init_op = tf.global_variables_initializer()  # name="init_op"

        if self.debug:
            endtime = time.time()
            print("Compilation time: {}".format(endtime - starttime))

            # count parameters
            self.num_weights = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print("total number of trainable weights : {}".format(self.num_weights))

    def init_session(self, session=None):
        if session is not None:
            # we are in an interactive notebook
            # session is passed to us
            self._sess = session
        else:
            config = tf.ConfigProto(device_count={"CPU": 8}, intra_op_parallelism_threads=8,
                                    inter_op_parallelism_threads=8)
            config.gpu_options.allow_growth = True

            # start tf session
            self._sess = tf.Session(config=config)

        self._sess.run(self.init_op)

    def init_trainingutils(self, save_dir=None):
        """
        if save_dir is not provided, summary, graph, and parameters (and everything really) will not be saved
        usually this is good if we are in an interactive environment just to play and test code
        """
        self.save_dir = save_dir
        if save_dir is not None:
            # construct tensorboard writers
            self.train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'summaries', 'train'))
            self.validation_writer = tf.summary.FileWriter(os.path.join(save_dir, 'summaries', 'validation'))
            self.test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'summaries', 'test'))

            # saving utils
            self._saver = tf.train.Saver(max_to_keep=1)

            def _save_model_params(epoch):

                path = os.path.join(save_dir, "saved_models", "model.ckpt")
                print(path)
                self._saver.save(self._sess, os.path.join(save_dir, "saved_models", "model.ckpt"), global_step=epoch,
                                 write_meta_graph=False)

            self.save_model_params = _save_model_params

        # Debug: if any nodes are added to graph, exception is raised
        tf.get_default_graph().finalize()

    def load_last_checkpoint(self):

        checkpoints_filenames = self._saver.last_checkpoints  # A list of checkpoint filenames, sorted from oldest to newest.
        self._saver.restore(self._sess, checkpoints_filenames[-1])

    def final_test_writer(self,log_likelihood):
        print(log_likelihood, file=open(os.path.join(self.save_dir, 'summaries', 'final_test'), "w"))

    def print_sumweights(self):
        """
        If the model has a SPN, print the sum weights
        :return:
        """
        variables_names = [v.name for v in tf.trainable_variables() if "weights" in v.name]
        values = self._sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}; Shape: {}; Value: {}".format(k, v.shape, v))


    def fit_earlystopping(self, datagraph, learning_rate, validation_period=None, maxStrikes=5, moving_average=None):
        """
        heuristic:
            evaluate on validation set every N batches
            save the model whenever validation set achieves highest score thus far
            when validation set performance decreases for maxStrikes consecutively, training stops
        :param datagraph:
        :param learning_rate:
        :param validation_period: int
            number of training batches before model is evaluated on validation set.
            usually set to the number of batches in an epoch
        :param maxStrikes:
        :return:
        """

        # counters and trackers
        strikes = 0
        numbatch = 0
        numepoch = 0
        max_validation = -np.inf

        training_curve = []
        validation_curve = []
        testing_curve = []

        start_time=time.time()
        while True:

            ############
            # Training #
            ############

            datagraph.switch2trainds(self._sess)
            for i in range(validation_period):
                opt, train_ll, summary_buffer = self._sess.run(
                    (self.train_op, self.mean_loglikelihood, self.summary_op),
                    feed_dict={self.learning_rate_ph: learning_rate})

                # if numbatch % 10 == 0:
                #     print(numbatch, str(train_ll))

                if self.save_dir:
                    self.train_writer.add_summary(summary_buffer, numbatch)

                training_curve.append(train_ll)

                numbatch += 1

            ##############
            # Validation #
            ##############

            datagraph.switch2valds(self._sess)
            val_lls = []
            for val_batch in range(10):
                val_ll, summary_buffer = self._sess.run((self.mean_loglikelihood, self.summary_op))
                val_lls.append(val_ll)
                if self.save_dir:
                    self.validation_writer.add_summary(summary_buffer, numbatch)
            validation_ll = np.mean(val_lls)
            print(numepoch, "Validation Log Likelihood " + str(validation_ll))

            # validation_ll, summary_buffer = self._sess.run((self.mean_loglikelihood, self.summary_op))
            # if self.save_dir:
            #     self.validation_writer.add_summary(summary_buffer, numbatch)
            # validation_curve.append(validation_ll)

            datagraph.switch2testds(self._sess)
            t_lls = []
            for test_batch in range(10):
                t_ll, summary_buffer = self._sess.run((self.mean_loglikelihood, self.summary_op))
                t_lls.append(t_ll)
                if self.save_dir:
                    self.test_writer.add_summary(summary_buffer, numbatch)

            test_ll = np.mean(t_lls)
            print(numepoch, "Test Log Likelihood " + str(test_ll))
            testing_curve.append(test_ll)

            if moving_average:
                if len(validation_curve) >= moving_average:
                    validation_ll_cmp = np.mean(validation_curve[-moving_average:])
                else:
                    validation_ll_cmp = np.mean(validation_curve)
            else:
                validation_ll_cmp = validation_ll

            if validation_ll_cmp > max_validation:
                print("Found Max validation! Saving model params!")
                max_validation = validation_ll_cmp
                if self.save_dir:
                    self.save_model_params(numbatch)
                strikes = 0
            else:
                strikes += 1
                print("Strike: {}".format(strikes))

            if strikes > maxStrikes:
                break

            numepoch += 1

        ###########
        # Testing #
        ###########

        if self.save_dir:
            datagraph.switch2testds(self._sess)
            self.load_last_checkpoint()
            t_lls = []
            for test_batch in range(10):
                t_ll = self._sess.run(self.mean_loglikelihood)
                t_lls.append(t_ll)
            test_ll = np.mean(t_lls)
            self.final_test_writer(test_ll)

        self.print_sumweights()

        output = {}
        output["num_weights"] = self.num_weights
        output["training_curve"] = training_curve
        output["validation_curve"] = validation_curve
        output["test_curve"] = testing_curve
        output["max_validation"] = max_validation
        output["test_ll"] = test_ll
        output["wall_time"] = time.time() - start_time
        output["start_time"] = start_time
        output["end_time"] = time.time()

        return output


