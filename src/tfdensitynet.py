import tensorflow as tf
from src.tfdistributions import *


# another way to implement a struct
class DenseNetArchitecture(object):
    def __init__(self, n_hidden=(20, 20), n_output=16, transfer = tf.nn.softplus):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.transfer_fct = transfer


class MLP(object):
    def __init__(self, name, n_hidden, n_output, transfer_fn):
        self.transfer_fn = transfer_fn
        self.name = name
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.hidden_kernel = []
        self.hidden_bias = []

        self.built = False

    def build(self, n_input):

        with tf.variable_scope(self.name):
            prev_n = n_input
            for i, n_layer in enumerate(self.n_hidden):
                self.hidden_kernel.append(tf.get_variable(name="hid_k_" + str(i),
                                                          shape=(prev_n, n_layer),
                                                          initializer=tf.contrib.layers.variance_scaling_initializer(
                                                              factor=1.),
                                                          regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                                          trainable=True)
                                          )
                self.hidden_bias.append(tf.get_variable(name="hid_b_" + str(i),
                                                        shape=(n_layer),
                                                        initializer=tf.zeros_initializer(),
                                                        regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                                        trainable=True)
                                        )
                prev_n = n_layer

            self.mean_kernel = tf.get_variable(name="mean_k",
                                               shape=(prev_n, self.n_output),
                                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.),
                                               regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                               trainable=True)
            self.mean_bias = tf.get_variable(name="mean_b",
                                             shape=(self.n_output),
                                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.),
                                             regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                             trainable=True)

        self.built = True

    def __call__(self, x):
        if not self.built:
            self.build(int(x.shape[-1]))
        prev_activation = x
        with tf.variable_scope(self.name):
            for kernel, bias in zip(self.hidden_kernel, self.hidden_bias):
                prev_activation = self.transfer_fn(tf.tensordot(prev_activation, kernel, [[-1], [0]]) + bias)

            output_mean = tf.tensordot(prev_activation, self.mean_kernel, [[-1], [0]]) + self.mean_bias
            return output_mean


class MLP2(object):
    def __init__(self, name, n_hidden, n_output, transfer_fn):
        self.transfer_fn = transfer_fn
        self.name = name
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.built = False

    def build(self, x):
        self.layers = []
        with tf.variable_scope(self.name):

            for i, nnodes in enumerate(self.n_hidden):
                layer = tf.layers.Dense(nnodes, activation=self.transfer_fn,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        name="hid_" + str(i),
                                       )
                self.layers.append(layer)


            output_layer = tf.layers.Dense(self.n_output, activation=None,
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           name="out",
                                          )
            self.layers.append(output_layer)

        self.built = True


    def __call__(self, x):
        if not self.built:
            self.build(x)
        prev_activation = x
        with tf.variable_scope(self.name):
            for layer in self.layers:
                prev_activation = layer(prev_activation)
        return prev_activation

class RandomNormal(object):
    def __init__(self, k_samples, nz):
        self.k_samples = k_samples
        self.nz = nz
        self.built = False

    def build(self, x):
        # need to know the batch size
        self.zs = tf.random_normal(shape=(self.k_samples, tf.shape(x)[0], self.nz), dtype=tf.float32)
        self.built = True

    def __call__(self, x):
        if not self.built:
            self.build(x)
        return self.zs

class DenseNetABC(object):
    """
    Abstract based class of DenseNet.
    Needed because there are multiple variants of DensityNet depending on the distribution type.
    Follows Factory Method Design Pattern where child class overrides a method relevant to construction of the object

    Child classes:
        BernoulliDenseNet
        GaussianDenseNet
        BinomialDenseNet
    """

    def __init__(self, architecture, k_samples=50, nz=1, zs=None):
        """

        :param architecture: of MLP
        :param k_samples: number of samples to do monte carlo integration of latent space
        :param nz: dimension of latent space
        :param zs: `Tensor` samples from latent space. if this argument is used, nz is ignored
        """
        self.k_samples = k_samples
        self.architecture = architecture
        self.nz = nz
        # samples from prior p(z)
        self.zs = zs

        self.built = False

    def build(self, x):
        with tf.variable_scope("densenet"):
            if self.zs is None:
                self.zs = RandomNormal(self.k_samples, self.nz)(x)
            # conditional p(x|z)
            self._create_conditional(self.zs, self.architecture)
        self.built = True

    def _create_conditional(self, zs, architecture):
        """
        compute parameters of p(x|z)
        :param zs: `Tensor`
        :param architecture:
        :return None

        this function returns None because the parameterization of p(z|x) may vary depending on distribution type
        """
        raise NotImplementedError


class BernoulliDenseNet(DenseNetABC):

    def __init__(self, architecture=DenseNetArchitecture(), k_samples=50, nz=1, zs=None):
        super(BernoulliDenseNet, self).__init__(architecture, k_samples, nz, zs)

    def _create_conditional(self, zs, architecture):
        n_hidden = architecture.n_hidden
        n_output = architecture.n_output
        transfer_fn = architecture.transfer_fct

        x_output_mean = MLP2("mlp", n_hidden, n_output, transfer_fn)(zs)
        # easy way to ensure that output is between 0 and 1
        self.x_reconstr_means = tf.nn.sigmoid(x_output_mean)  # shape == (k_samples, batch, n_output)

    def __call__(self, x):
        if not self.built:
            self.build(x)
        with tf.variable_scope("densenet"):
            x_expand = tf.expand_dims(x, axis=0)  # shape == (1, batch, n_input)
            x_tiled = tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape == (k_samples, batch, n_input)

            # log p(x|z)
            lpxgivenz = tf.reduce_sum(bernoulli_logprob(x_tiled, self.x_reconstr_means),
                                      axis=2)  # shape == (k_samples, batch)
            # print(lpxgivenz.name)
            # print_op = tf.print(tf.count_nonzero(tf.greater(lpxgivenz, 0.)))
            # with tf.control_dependencies([print_op, tf.assert_non_positive(lpxgivenz)]):
            # Monte carlo importance weighted integration
            lpx = tf.reduce_logsumexp(lpxgivenz, axis=0) - np.log(self.k_samples)  # shape == (batch)
            return lpx


class GaussianDenseNet(DenseNetABC):

    def __init__(self, architecture=DenseNetArchitecture(), k_samples=50, nz=1, zs=None):
        super(GaussianDenseNet, self).__init__(architecture, k_samples, nz, zs)

    def _create_conditional(self, zs, architecture):
        n_hidden = architecture.n_hidden
        n_output = architecture.n_output
        transfer_fn = architecture.transfer_fct

        with tf.variable_scope("conditional"):
            x_output_mean = MLP2("mean_mlp", n_hidden, n_output, transfer_fn)(zs)
            x_output_sigma = MLP2("sigma_mlp", n_hidden, n_output, transfer_fn)(zs)

            # easy way to ensure that output is between 0 and 1
            self.x_reconstr_means = tf.nn.sigmoid(x_output_mean)  # shape == (k_samples, batch, n_output)
            # infusing context knowledge that variance cannot be greater than 1 (actually, 1/sqrt(6))
            # self.x_reconstr_sigmas = tf.nn.sigmoid(x_output_sigma)  # shape == (k_samples, batch, n_output)
            self.x_reconstr_inv_sigma_sq = 1. + tf.square(x_output_sigma)  # shape == (k_samples, batch, n_output)

            # TODO: add checks for constructed parameters

    def __call__(self, x):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            x_expand = tf.expand_dims(x, axis=0)  # shape == (1, batch, n_input)
            x_tiled = tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape == (k_samples, batch, n_input)

            # p(x|z)
            samples = gaussian_logprob2(x_tiled, self.x_reconstr_means, self.x_reconstr_inv_sigma_sq)
            lpxgivenz = tf.reduce_sum(samples, axis=2)  # shape == (k_samples, batch)

            # Monte carlo importance weighted integration
            lpx = tf.reduce_logsumexp(lpxgivenz, axis=0) - np.log(self.k_samples)  # shape == (batch)
            return lpx


class BinomialDenseNet(DenseNetABC):

    def __init__(self, architecture=DenseNetArchitecture(), k_samples=50, nz=1, zs=None):
        super(BinomialDenseNet, self).__init__(architecture, k_samples, nz, zs)

    def _create_conditional(self, zs, architecture):
        n_hidden = architecture.n_hidden
        n_output = architecture.n_output
        transfer_fn = architecture.transfer_fct

        with tf.variable_scope("conditional"):
            x_output_mean = MLP2("mlp", n_hidden, n_output, transfer_fn)(zs)
            # easy way to ensure that output is between 0 and 1
            self.x_reconstr_means = tf.nn.sigmoid(x_output_mean)  # shape == (k_samples, batch, n_output)

            # TODO: add checks for constructed parameters

    def __call__(self, x):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            x_expand = tf.expand_dims(x, axis=0)  # shape == (1, batch, n_input)
            x_tiled = 255 * tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape == (k_samples, batch, n_input)

            # p(x|z)
            lpxgivenz = tf.reduce_sum(binomial_logprob(x_tiled, self.x_reconstr_means, 255),
                                      axis=2)  # shape == (k_samples, batch)

            with tf.control_dependencies([tf.assert_non_positive(lpxgivenz)]):
                # Monte carlo importance weighted integration
                lpx = tf.reduce_logsumexp(lpxgivenz, axis=0) - np.log(self.k_samples)  # shape == (batch)
                return lpx


class InferenceNet(object):
    def __init__(self, recog_archi, k_samples, sigma_param="1sq"):
        self.sigma_param = sigma_param
        self.recog_archi = recog_archi
        self.k_samples=k_samples

        self.built = False

    def build(self):
        n_hidden = self.recog_archi.n_hidden
        n_output = self.recog_archi.n_output
        transfer_fn = self.recog_archi.transfer_fct

        self.mean_mlp = MLP2("var_mean_mlp", n_hidden, n_output, transfer_fn)
        self.sigma_mlp = MLP2("var_sigma_mlp", n_hidden, n_output, transfer_fn)

        self.built = True

    def __call__(self, x):
        if not self.built:
            self.build()
        z_mu = self.mean_mlp(x)

        if self.sigma_param == "1sq":
            z_inv_sigma_sq = 1. + tf.square(self.sigma_mlp(x))
        elif self.sigma_param == "sq":
            z_inv_sigma_sq = tf.square(self.sigma_mlp(x))
        elif self.sigma_param == "exp":
            z_inv_sigma_sq = tf.exp(self.sigma_mlp(x))
        else:
            raise ValueError

        z_dim = int(z_mu.shape[1])
        assert int(z_mu.shape[1]) == int(z_inv_sigma_sq.shape[1])
        # batch_size = int(z_mu.shape[0])
        batch_size = tf.shape(z_mu)[0]
        # assert int(z_mu.shape[0]) == int(z_inv_sigma_sq.shape[0])
        shape = (self.k_samples, batch_size, z_dim)

        zs = gaussian_sample(shape, z_mu, z_inv_sigma_sq)

        return z_mu, z_inv_sigma_sq, zs


class RandomGaussian(object):
    def __init__(self, k_samples, z_mu, z_inv_sigma_sq):
        self.k_samples = k_samples
        self.z_mu = z_mu
        self.z_inv_sigma_sq = z_inv_sigma_sq
        self.built = False

    def build(self):
        # need to know the batch size
        z_dim = int(self.z_mu.shape[1])
        assert int(self.z_mu.shape[1]) == int(self.z_inv_sigma_sq.shape[1])
        # batch_size = int(z_mu.shape[0])
        batch_size = tf.shape(self.z_mu)[0]
        # assert int(z_mu.shape[0]) == int(z_inv_sigma_sq.shape[0])
        shape = (self.k_samples, batch_size, z_dim)

        self.zs = gaussian_sample(shape, self.z_mu, self.z_inv_sigma_sq)

        self.built = True

    def __call__(self):
        if not self.built:
            self.build()
        return self.zs

class IWAEABC(object):
    """
    Abstract base class of importance weighted autoencoder
    Uses factory method design pattern to let child class influence construction of object

    Child Classes:
        BernoulliIWAE
        GaussianIWAE
        BinomialIWAE
    """

    def __init__(self, recog_architecture, gener_architecture, k_samples, sigma_param="1sq", qz_params=None):
        """

        :param recog_architecture:
        :param gener_architecture:
        :param k_samples:
        :param qz_params: z_mu, z_inv_sigma_sq, zs
                if provided, this model will not build its own inference network
        """

        assert isinstance(gener_architecture, DenseNetArchitecture)

        self.recog_architecture = recog_architecture
        self.gener_architecture = gener_architecture
        # using Variable so that I can experiment on effect of increasing K
        self.k_samples = tf.Variable(k_samples, trainable=False, dtype=tf.int32)
        self.qz_params = qz_params
        self.sigma_param = sigma_param

        self.built = False

    def build(self, x):

        self._create_variational(x)
        self._create_generator()
        self.built = True

    def _create_variational(self, x):
        """
        Compute 4 quantities from a diagonal gaussian variational distribution
            - z_mu : mean of the variational posterior
            - z_inv_sigma_sq : precision of the variational posterior
            - zs : samples from the variational posterior
            - lpz : log p(z) log likelihood of the samples according to the prior
        """

        # if qz_params is not provided, we will build our own inference network
        if self.qz_params is None:
            assert isinstance(self.recog_architecture, DenseNetArchitecture)
            self.z_mu, self.z_inv_sigma_sq, self.zs = InferenceNet(self.recog_architecture, self.k_samples, self.sigma_param)(x)
        else:
            self.z_mu, self.z_inv_sigma_sq, self.zs = self.qz_params

        # log p(z)
        self.lpz = tf.reduce_sum(standardnormal_logprob(self.zs), axis=2)  # shape == (k_samples, batch)

    def _create_generator(self):
        n_hidden = self.gener_architecture.n_hidden
        n_output = self.gener_architecture.n_output
        transfer_fn = self.gener_architecture.transfer_fct

        self.gen_mean_mlp = MLP2("gen_mlp", n_hidden, n_output, transfer_fn)


class BernoulliIWAE(IWAEABC):

    def __init__(self, recog_architecture, gener_architecture, k_samples=5, sigma_param="1sq", qz_params=None):
        super(BernoulliIWAE, self).__init__(recog_architecture, gener_architecture, k_samples, sigma_param=sigma_param,
                                            qz_params=qz_params)

    def __call__(self, x, average=True):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            z_mu, z_inv_sigma_sq = self.z_mu, self.z_inv_sigma_sq
            zs, lpz = self.zs, self.lpz

            # log p(x|z)
            x_expand = tf.expand_dims(x, axis=0)  # shape = (1, batch, n_input)
            x_tiled = tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape = (k_samples, batch, n_input)
            self.x_reconstr_means = tf.nn.sigmoid(self.gen_mean_mlp(zs))
            lpxgivenz = tf.reduce_sum(bernoulli_logprob(x_tiled, self.x_reconstr_means),
                                      axis=2)  # shape = (k_samples, batch)
            self.lpxgivenz = lpxgivenz

            # log p(x,z) = log p(x|z) * p(z)
            log_numerator = tf.add(lpxgivenz, lpz)  # shape == (k_samples, batch,)

            # log q(z|x)
            log_denominator = tf.reduce_sum(gaussian_logprob2(zs, z_mu, z_inv_sigma_sq),
                                            axis=2)  # shape == (k_samples, batch,)

            # importance weighting of samples
            likelihood_samples = tf.subtract(log_numerator, log_denominator)  # shape == (k_samples, batch,)
            if average:
                avg_likelihood = tf.reduce_logsumexp(likelihood_samples, axis=0) - tf.log(
                    tf.cast(self.k_samples, tf.float32))  # shape == (batch,)

                return avg_likelihood
            else:
                return likelihood_samples


class BinomialIWAE(IWAEABC):

    def __init__(self, recog_architecture, gener_architecture, k_samples=5, sigma_param="1sq", qz_params=None):
        super(BinomialIWAE, self).__init__(recog_architecture, gener_architecture, k_samples, sigma_param=sigma_param,
                                           qz_params=qz_params)

    def __call__(self, x, average=True):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            z_mu, z_inv_sigma_sq = self.z_mu, self.z_inv_sigma_sq
            zs, lpz = self.zs, self.lpz

            # log p(x|z)
            x_expand = tf.expand_dims(x, axis=0)  # shape = (1, batch, n_input)
            x_tiled = 255. * tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape = (k_samples, batch, n_input)

            # log p(x|z)
            x_reconstr_means = tf.nn.sigmoid(self.gen_mean_mlp(zs))
            self.x_reconstr_means = x_reconstr_means
            lpxgivenz = tf.reduce_sum(binomial_logprob(x_tiled, x_reconstr_means, 255.),
                                      axis=2)  # shape = (k_samples, batch)

            # log p(x,z) = log p(x|z) * p(z)
            log_numerator = tf.add(lpxgivenz, lpz)  # shape == (k_samples, batch,)

            # log q(z|x)
            log_denominator = tf.reduce_sum(gaussian_logprob2(zs, z_mu, z_inv_sigma_sq),
                                            axis=2)  # shape == (k_samples, batch,)

            # importance weighting of samples
            likelihood_samples = tf.subtract(log_numerator, log_denominator)  # shape == (k_samples, batch,)
            if average:
                avg_likelihood = tf.reduce_logsumexp(likelihood_samples, axis=0) - tf.log(
                    tf.cast(self.k_samples, tf.float32))  # shape == (batch,)

                return tf.check_numerics(avg_likelihood, "binomialVAE")
            else:
                return likelihood_samples



class GaussianIWAE(IWAEABC):

    def __init__(self, recog_architecture, gener_architecture, k_samples=5, sigma_param="1sq", qz_params=None):
        super(GaussianIWAE, self).__init__(recog_architecture, gener_architecture, k_samples, sigma_param=sigma_param,
                                           qz_params=qz_params)

    def _create_generator(self):
        with tf.variable_scope("gen_net"):
            n_hidden = self.gener_architecture.n_hidden
            n_output = self.gener_architecture.n_output
            transfer_fn = self.gener_architecture.transfer_fct

            self.gen_mean_mlp = MLP2("gen_mlp", n_hidden, n_output, transfer_fn)
            self.gen_var_mlp = MLP2("var_mlp", n_hidden, n_output, transfer_fn)

    def __call__(self, x, average=True):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            z_mu, z_inv_sigma_sq = self.z_mu, self.z_inv_sigma_sq
            zs, lpz = self.zs, self.lpz

            x_expand = tf.expand_dims(x, axis=0)  # shape = (1, batch, n_input)
            x_tiled = tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape = (k_samples, batch, n_input)

            # log p(x|z)
            x_reconstr_means = tf.nn.sigmoid(self.gen_mean_mlp(zs))

            self.x_reconstr_means = x_reconstr_means

            x_reconstr_inv_sigma_sq = 1 + tf.square(self.gen_var_mlp(zs))

            lpxgivenz = tf.check_numerics(
                tf.reduce_sum(gaussian_logprob2(x_tiled, x_reconstr_means, x_reconstr_inv_sigma_sq), axis=2),
                "p(x|z) has problem")  # shape = (k_samples, batch)

            # log p(x,z) = log p(x|z) * p(z)
            log_numerator = tf.add(lpxgivenz, lpz)  # shape == (k_samples, batch,)

            # log q(z|x)
            log_denominator = tf.reduce_sum(gaussian_logprob2(zs, z_mu, z_inv_sigma_sq),
                                            axis=2)  # shape == (k_samples, batch,)

            # importance weighting of samples
            likelihood_samples = tf.subtract(log_numerator, log_denominator)  # shape == (k_samples, batch,)

            if average:
                avg_likelihood = tf.check_numerics(tf.reduce_logsumexp(likelihood_samples, axis=0),
                                                   "logsumexp has problem") - tf.log(
                    tf.cast(self.k_samples, tf.float32))  # shape == (batch,)

                return avg_likelihood
            else:
                return likelihood_samples


class VAEABC(object):
    """
    Abstract base class of variational autoencoder
    Uses factory method design pattern to let child class influence construction of object

    Child Classes:
        BernoulliIWAE
        GaussianIWAE
        BinomialIWAE
    """

    def __init__(self, recog_architecture, gener_architecture, qz_params=None, zs=None):
        """

        :param recog_architecture:
        :param gener_architecture:
        :param k_samples:
        :param qz_params: parameters from inference network.
                if provided, this model will not build its own inference network
        :param zs: samples from the variation distribution.
                if provided, this model will not build its own inference network nor sample from it
                if provided, z_params must also be provided (it is needed to compute the ELBO )
        """

        assert isinstance(gener_architecture, DenseNetArchitecture)

        self.recog_architecture = recog_architecture
        self.gener_architecture = gener_architecture
        # using Variable so that I can experiment on effect of increasing K
        self.qz_params = qz_params
        self.zs = zs

        self.built = False

    def build(self, x):

        self._create_variational(x)
        self._create_generator()
        self.built = True

    def _create_variational(self, x):
        """

        """

        if self.zs is None:
            if self.qz_params is None:
                assert isinstance(self.recog_architecture, DenseNetArchitecture)

                n_hidden = self.recog_architecture.n_hidden
                n_output = self.recog_architecture.n_output
                transfer_fn = self.recog_architecture.transfer_fct

                mean_mlp = MLP2("var_mean_mlp", n_hidden, n_output, transfer_fn)
                sigma_mlp = MLP2("var_sigma_mlp", n_hidden, n_output, transfer_fn)

                self.z_mu = mean_mlp(x)
                self.log_z_sigma_sq = sigma_mlp(x)
            else:
                self.z_mu, self.log_z_sigma_sq = self.qz_params

            self.z_sigma_sq = tf.exp(self.log_z_sigma_sq)

            # sample z
            # preparing shape of samples
            z_dim = int(self.z_mu.shape[1])
            assert int(self.z_mu.shape[1]) == int(self.z_sigma_sq.shape[1])
            # batch_size = int(self.z_mu.shape[0])
            batch_size = tf.shape(self.z_mu)[0]
            # assert int(self.z_mu.shape[0]) == int(self.z_inv_sigma_sq.shape[0])

            shape = (batch_size, z_dim)

            self.zs = gaussian_sample2(shape, self.z_mu, tf.sqrt(self.z_sigma_sq))

        else:
            if self.qz_params is None:
                raise Exception("z_params must be provided if zs is provided. "
                                "z_params are (mu and inv_sigma_sq) of a multivariate gaussian")
            else:
                self.z_mu, self.log_z_sigma_sq = self.qz_params

    def _create_generator(self):
        n_hidden = self.gener_architecture.n_hidden
        n_output = self.gener_architecture.n_output
        transfer_fn = self.gener_architecture.transfer_fct

        self.gen_mean_mlp = MLP2("gen_mlp", n_hidden, n_output, transfer_fn)


class BernoulliVAE(VAEABC):

    def __init__(self, recog_architecture, gener_architecture, qz_params=None, zs=None):
        super(BernoulliVAE, self).__init__(recog_architecture, gener_architecture, qz_params=qz_params, zs=zs)

    def __call__(self, x):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            self.x_reconstr_mean = tf.nn.sigmoid(self.gen_mean_mlp(self.zs))

            reconstr_elbo = tf.reduce_sum(bernoulli_logprob(x, self.x_reconstr_mean), axis=1)

            latent_elbo = 0.5 * tf.reduce_sum(-1 - self.log_z_sigma_sq
                                              + tf.square(self.z_mu)
                                              + self.z_sigma_sq, axis=1)

            # print_op = tf.print(tf.reduce_mean(latent_elbo), latent_elbo.shape, tf.reduce_mean(reconstr_elbo), reconstr_elbo.shape)
            # with tf.control_dependencies([print_op]):
            #     elbo = reconstr_elbo - latent_elbo
            elbo = reconstr_elbo - latent_elbo
            return elbo
