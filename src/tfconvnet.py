from src.tflayer_leaf import TFLayer
from src.tfdistributions import *


class ConvDenseArchitecture(object):
    def __init__(self, filters=(16, 8), kernel_size=(3, 3), stride=(1,1), padding=('valid', 'valid'),
                 n_dense=(20, 16), transfer=tf.nn.softplus, latent_nz = 2):
        self.n_dense = n_dense
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.transfer_fct = transfer
        self.latent_nz = latent_nz


class ConvDense(object):
    def __init__(self, name, recog_archi):
        self.name = name
        self.recog_archi = recog_archi
        self.built = False

    def build(self):
        self.layers = []
        if self.recog_archi.convFirst:
            for f, k, s, p in zip(self.recog_archi.filters, self.recog_archi.kernel_size,
                                  self.recog_archi.stride, self.recog_archi.padding):
                self.layers.append(tf.layers.Conv2D(
                    filters=f,
                    kernel_size=k,
                    strides=s,
                    padding=p,
                    activation=self.recog_archi.transfer_fct)
                )

            for nnodes in self.recog_archi.n_dense:
                self.layers.append(tf.layers.Dense(units=nnodes,
                                                   activation=self.recog_archi.transfer_fn,
                                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                   ))

        else:
            for nnodes in self.recog_archi.n_dense:
                self.layers.append(tf.layers.Dense(units=nnodes,
                                                   activation=self.recog_archi.transfer_fn,
                                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                   ))

            for f, k, s, p in zip(self.recog_archi.filters, self.recog_archi.kernel_size,
                                  self.recog_archi.stride, self.recog_archi.padding):
                self.layers.append(tf.layers.Conv2DTranspose(
                    filters=f,
                    kernel_size=k,
                    strides=s,
                    padding=p,
                    activation=self.recog_archi.transfer_fct)
                )
        self.built = True

    def __call__(self, x):
        if not self.built:
            self.build()
        prev_activation = x
        with tf.variable_scope(self.name):
            for layer in self.layers:
                prev_activation = layer(prev_activation)
        return prev_activation


class InferenceNet_conv(object):
    def __init__(self, recog_archi, k_samples, sigma_param="1sq"):
        self.sigma_param = sigma_param
        self.k_samples = k_samples
        self.recog_archi = recog_archi

        self.built = False

    def build(self):
        self.mean_convdense = ConvDense("var_mean_cd", self.recog_archi)
        self.sigma_convdense = ConvDense("var_sigma_cd", self.recog_archi)

        self.built = True

    def __call__(self, x):
        if not self.built:
            self.build()

        z_mu = self.mean_convdense(x)

        if self.sigma_param == "1sq":
            z_inv_sigma_sq = 1. + tf.square(self.sigma_convdense(x))
        elif self.sigma_param == "sq":
            z_inv_sigma_sq = tf.square(self.sigma_convdense(x))
        elif self.sigma_param == "exp":
            z_inv_sigma_sq = tf.exp(self.sigma_convdense(x))
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


class IWAEABC_conv2(object):
    """
    Abstract base class of importance weighted autoencoder
    Uses factory method design pattern to let child class influence construction of object

    Child Classes:
        BernoulliIWAE
        GaussianIWAE
        BinomialIWAE
    """

    def __init__(self, architecture, k_samples, sigma_param="1sq"):

        assert isinstance(architecture, ConvDenseArchitecture)

        self.architecture = architecture

        # using Variable so that I can experiment on effect of increasing K
        self.k_samples = tf.Variable(k_samples, trainable=False, dtype=tf.int32)
        self.sigma_param = sigma_param

        self.built = False

    def build(self, x):

        self._create_variational(x)
        self._create_generator()
        self.built = True

    def _create_variational(self, x):
        """
        creates a mean and sigma network

        the k_samples dimension is created after the mean and sigma is predicted

        Compute 4 quantities from a diagonal gaussian variational distribution
            - z_mu : mean of the variational posterior
            - z_inv_sigma_sq : precision of the variational posterior
            - zs : samples from the variational posterior
            - lpz : log p(z) log likelihood of the samples according to the prior

        :param x: shape = (batch, nvars)
        """

        # if qz_params is not provided, we will build our own inference network
        assert isinstance(self.architecture, ConvDenseArchitecture)
        batch_size = int(x.shape[0])
        self.batch_size = batch_size

        ################
        # Mean network #
        ################

        mean_activation = x  # shape = (batch, vars)

        # gonna assume that the image is a square
        mean_activation = tf.reshape(mean_activation,
            (-1, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])), 1))  # shape = (batch, height, width, filter)

        for f, k, s, p in zip(self.architecture.filters, self.architecture.kernel_size,
                              self.architecture.stride, self.architecture.padding):
            mean_activation = tf.layers.Conv2D(
                filters=f,
                kernel_size=k,
                strides=s,
                padding=p,
                activation=self.architecture.transfer_fct)(mean_activation) # shape = (batch, height, width, filter)

        self.shape_between_sigma_conv_dense = mean_activation.shape  # shape = (batch, height, width, filter)
        mean_activation = tf.reshape(mean_activation, (batch_size, -1))

        for nnodes in self.architecture.n_dense:
            mean_activation = tf.layers.Dense(units=nnodes,
                                               activation=self.architecture.transfer_fct,
                                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                               )(mean_activation)

        # linear activation on the final activation layer
        z_mu = tf.layers.Dense(units=self.architecture.latent_nz,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          )(mean_activation)

        #################
        # Sigma network #
        #################

        sigma_activation = x  # shape = (batch, vars)
        sigma_activation = tf.reshape(sigma_activation,
            (-1, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))  # shape = (batch, height, width)

        for f, k, s, p in zip(self.architecture.filters, self.architecture.kernel_size,
                              self.architecture.stride, self.architecture.padding):
            sigma_activation = tf.layers.Conv2D(
                filters=f,
                kernel_size=k,
                strides=s,
                padding=p,
                activation=self.architecture.transfer_fct)(sigma_activation) # shape = (batch, height, width, filter)

        self.shape_between_sigma_conv_dense = sigma_activation.shape  # shape = (batch, height, width, filter)

        sigma_activation = tf.reshape(sigma_activation, (batch_size, -1))

        for nnodes in self.architecture.n_dense:
            sigma_activation = tf.layers.Dense(units=nnodes,
                                               activation=self.architecture.transfer_fct,
                                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                               )(sigma_activation)


        sigma_activation = tf.layers.Dense(units=self.architecture.latent_nz,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          )(sigma_activation)

        if self.sigma_param == "1sq":
            z_inv_sigma_sq = 1. + tf.square(sigma_activation)
        elif self.sigma_param == "sq":
            z_inv_sigma_sq = tf.square(sigma_activation)
        elif self.sigma_param == "exp":
            z_inv_sigma_sq = tf.exp(sigma_activation)
        else:
            raise ValueError

        #######################################
        # Sampling (reparameterization trick) #
        #######################################


        zs = gaussian_sample((self.k_samples, batch_size, self.architecture.latent_nz), z_mu, z_inv_sigma_sq)

        self.z_mu, self.z_inv_sigma_sq, self.zs = z_mu, z_inv_sigma_sq, zs

        # log p(z)
        self.lpz = tf.reduce_sum(standardnormal_logprob(self.zs), axis=2)  # shape == (k_samples, batch)

    def _create_generator(self):
        """
        for the discrete case

        unlike the variational network, the generative network has a k_sample dimension throughout
        :return:
        """
        mean_activation = self.zs  # shape = (k_samples, batch, nz)
        print(mean_activation.shape)
        for nnodes in reversed(self.architecture.n_dense):
            mean_activation = tf.layers.Dense(units=nnodes,
                                               activation=self.architecture.transfer_fct,
                                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                               )(mean_activation)
            print(mean_activation.shape)

        mean_activation = tf.reshape(mean_activation, self.shape_between_mean_conv_dense)  # shape = (k_samples, batch, height, width, filter)
        print(mean_activation.shape)

        for f, k, s, p in zip(reversed(self.architecture.filters), reversed(self.architecture.kernel_size),
                              reversed(self.architecture.stride), reversed(self.architecture.padding)):

            mean_activation = tf.layers.Conv2DTranspose(
                filters=f,
                kernel_size=k,
                s=s,
                padding=p,
                activation=self.architecture.transfer_fct)(mean_activation) # shape = (batch, height, width, filter)
            print(mean_activation.shape)

        mean_activation = tf.reshape(mean_activation, (self.batch_size, -1))
        print(mean_activation.shape)
        self.x_reconstr_means = tf.layers.Conv2DTranspose(filters=1,
                                        kernel_size=k,
                                        strides=s,
                                        padding=p,
                                         )(mean_activation)
        print(mean_activation.shape)

        self.x_reconstr_means = tf.nn.sigmoid(self.x_reconstr_means)

class IWAEABC_conv(object):
    """
    Abstract base class of importance weighted autoencoder
    Uses factory method design pattern to let child class influence construction of object

    Child Classes:
        BernoulliIWAE
        GaussianIWAE
        BinomialIWAE
    """

    def __init__(self, architecture, k_samples, sigma_param="1sq"):

        assert isinstance(architecture, ConvDenseArchitecture)

        self.architecture = architecture

        # using Variable so that I can experiment on effect of increasing K
        self.k_samples = tf.Variable(k_samples, trainable=False, dtype=tf.int32)
        self.sigma_param = sigma_param

        self.built = False

    def build(self, x):

        self._create_variational(x)
        self._create_generator()
        self.built = True

    def _create_variational(self, x):
        """
        creates a mean and sigma network

        the k_samples dimension is created after the mean and sigma is predicted

        Compute 4 quantities from a diagonal gaussian variational distribution
            - z_mu : mean of the variational posterior
            - z_inv_sigma_sq : precision of the variational posterior
            - zs : samples from the variational posterior
            - lpz : log p(z) log likelihood of the samples according to the prior

        :param x: shape = (batch, nvars)
        """

        # if qz_params is not provided, we will build our own inference network
        assert isinstance(self.architecture, ConvDenseArchitecture)

        ################
        # Conv network #
        ################
        self.batch_size = tf.shape(x)[0]



        activation = x  # shape = (batch, vars)
        print(activation.shape)
        # gonna assume that the image is a square
        activation = tf.reshape(activation,
            (self.batch_size, int(np.sqrt(int(x.shape[1]))), int(np.sqrt(int(x.shape[1]))), 1))  # shape = (batch, height, width, filter)

        print(activation.shape)

        for f, k, s, p in zip(self.architecture.filters, self.architecture.kernel_size,
                              self.architecture.stride, self.architecture.padding):
            activation = tf.layers.Conv2D(
                filters=f,
                kernel_size=k,
                strides=s,
                padding=p,
                activation=self.architecture.transfer_fct)(activation) # shape = (batch, height, width, filter)
            print(activation.shape)

        self.shape_between_conv_dense = activation.shape  # shape = (batch, height, width, filter)
        nnodes = int(self.shape_between_conv_dense[1]) * \
                 int(self.shape_between_conv_dense[2]) * int(self.shape_between_conv_dense[3])

        activation = tf.reshape(activation, (self.batch_size, nnodes))
        print(activation.shape)
        #################
        # Dense network #
        #################

        for nnodes in self.architecture.n_dense:
            activation = tf.layers.Dense(units=nnodes,
                                               activation=self.architecture.transfer_fct,
                                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                               )(activation)

            print(activation.shape)

        #################
        # mean output #
        #################

        # linear activation on the final activation layer
        z_mu = tf.layers.Dense(units=self.architecture.latent_nz,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          )(activation)

        print(z_mu.shape)

        #################
        # sigma output #
        #################

        z_inv_sigma_sq = tf.layers.Dense(units=self.architecture.latent_nz,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          )(activation)

        if self.sigma_param == "1sq":
            z_inv_sigma_sq = 1. + tf.square(z_inv_sigma_sq)
        elif self.sigma_param == "sq":
            z_inv_sigma_sq = tf.square(z_inv_sigma_sq)
        elif self.sigma_param == "exp":
            z_inv_sigma_sq = tf.exp(z_inv_sigma_sq)
        else:
            raise ValueError

        #######################################
        # Sampling (reparameterization trick) #
        #######################################


        zs = gaussian_sample((self.k_samples, self.batch_size, self.architecture.latent_nz), z_mu, z_inv_sigma_sq)
        print(self.k_samples, self.batch_size, self.architecture.latent_nz)
        print("zs", zs.shape)
        self.z_mu, self.z_inv_sigma_sq, self.zs = z_mu, z_inv_sigma_sq, zs

        # log p(z)
        self.lpz = tf.reduce_sum(standardnormal_logprob(self.zs), axis=2)  # shape == (k_samples, batch)

    def _create_generator(self):
        """
        for the discrete case

        unlike the variational network, the generative network has a k_sample dimension throughout
        :return:
        """

        #################
        # Dense network #
        #################

        activation = self.zs  # shape = (k_samples, batch, nz)
        print(activation.shape)
        # assert int(activation.shape[0]) == self.k_samples
        # assert int(activation.shape[1]) == self.batch_size

        nnodes = int(self.shape_between_conv_dense[1]) * \
                 int(self.shape_between_conv_dense[2]) * int(self.shape_between_conv_dense[3])
        activation = tf.layers.Dense(units=nnodes,
                                       activation=self.architecture.transfer_fct,
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       )(activation)

        print(activation.shape)
        # conv2Dtranspose cannot cannot an extra k_sample dimension
        activation = tf.reshape(activation, (self.k_samples*self.batch_size,
                                             int(self.shape_between_conv_dense[1]),
                                             int(self.shape_between_conv_dense[2]),
                                             int(self.shape_between_conv_dense[3])))  # shape = (k_samples* batch, height, width, filter)
        print(activation.shape)

        #################
        # Conv network #
        #################

        for f, k, s, p in zip(reversed(self.architecture.filters[:-1]), reversed(self.architecture.kernel_size),
                              reversed(self.architecture.stride), reversed(self.architecture.padding)):
            activation = tf.layers.Conv2DTranspose(
                filters=f,
                kernel_size=k,
                strides=s,
                padding=p,
                activation=self.architecture.transfer_fct)(activation) # shape = (k_samples*batch, height, width, filter)
            print(activation.shape)

        print((self.k_samples, self.batch_size,
                int(activation.shape[1]), int(activation.shape[2]),
               int(activation.shape[3])))
        activation = tf.layers.Conv2DTranspose(filters=1,
                                        kernel_size=k,
                                        strides=s,
                                        padding=p,
                                         )(activation)


        print((self.k_samples, self.batch_size,
                                             int(activation.shape[1]), int(activation.shape[2]), 1))


        activation = tf.reshape(activation, (self.k_samples, self.batch_size,
                                             int(activation.shape[1]), int(activation.shape[2]), 1))
        print(activation.shape)
        # activation = tf.Print(activation, [tf.reduce_max(activation),tf.reduce_min(activation)])

        activation = tf.nn.sigmoid(activation)
        # activation = tf.Print(activation, [tf.reduce_max(activation),tf.reduce_min(activation)])

        self.x_reconstr_means = tf.reshape(activation, (self.k_samples, self.batch_size, -1))
        print(activation.shape)


class BernoulliIWAE_conv(IWAEABC_conv):

    def __init__(self, architecture, k_samples=5, sigma_param="1sq",):
        super(BernoulliIWAE_conv, self).__init__(architecture, k_samples,
                                                 sigma_param=sigma_param)

    def __call__(self, x, average=True):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            z_mu, z_inv_sigma_sq = self.z_mu, self.z_inv_sigma_sq
            zs, lpz = self.zs, self.lpz

            # log p(x|z)
            x_expand = tf.expand_dims(x, axis=0)  # shape = (1, batch, n_input)
            x_expand = tf.expand_dims(x_expand, axis=3)  # shape = (1, batch, n_input, n_filter=1)
            x_tiled = tf.tile(x_expand, multiples=(self.k_samples, 1, 1, 1))  # shape = (k_samples, batch, n_input)

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


class BinomialIWAE_conv(IWAEABC_conv):

    def __init__(self, architecture, k_samples=5, sigma_param="1sq"):
        super(BinomialIWAE_conv, self).__init__(architecture, k_samples,
                                                sigma_param=sigma_param)

    def __call__(self, x, average=True):
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            z_mu, z_inv_sigma_sq = self.z_mu, self.z_inv_sigma_sq
            zs, lpz = self.zs, self.lpz

            # log p(x|z)
            x_expand = tf.expand_dims(x, axis=0)  # shape = (1, batch, n_input)
            # x_expand = tf.expand_dims(x_expand, axis=3)  # shape = (1, batch, n_input, n_filter=1)
            x_tiled = 255. * tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape = (k_samples, batch, n_input)

            # log p(x|z)
            lpxgivenz = tf.reduce_sum(binomial_logprob(x_tiled, self.x_reconstr_means, 255.),
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


class GaussianIWAE_conv(IWAEABC_conv):

    def __init__(self, architecture, k_samples=5, sigma_param="1sq"):
        super(GaussianIWAE_conv, self).__init__(architecture, k_samples,
                                                sigma_param=sigma_param)

    def _create_generator(self):
        with tf.variable_scope("gen_net"):
            # self.gen_mean_cd = ConvDense("gen_mean_cd", self.architecture)
            # self.gen_var_cd = ConvDense("gen_sigma_cd", self.architecture)

            #################
            # Dense network #
            #################

            activation = self.zs  # shape = (k_samples, batch, nz)
            print(activation.shape)
            # assert int(activation.shape[0]) == self.k_samples
            # assert int(activation.shape[1]) == self.batch_size

            nnodes = int(self.shape_between_conv_dense[1]) * \
                     int(self.shape_between_conv_dense[2]) * int(self.shape_between_conv_dense[3])

            activation = tf.layers.Dense(units=nnodes,
                                         activation=self.architecture.transfer_fct,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         )(activation)
            print(activation.shape)

            # conv2Dtranspose cannot cannot an extra k_sample dimension
            activation = tf.reshape(activation, (self.k_samples * self.batch_size,
                                                 int(self.shape_between_conv_dense[1]),
                                                 int(self.shape_between_conv_dense[2]),
                                                 int(self.shape_between_conv_dense[
                                                         3])))  # shape = (k_samples* batch, height, width, filter)
            print(activation.shape)

            #################
            # Conv network #
            #################

            for f, k, s, p in zip(reversed(self.architecture.filters[:-1]), reversed(self.architecture.kernel_size),
                                  reversed(self.architecture.stride), reversed(self.architecture.padding)):
                activation = tf.layers.Conv2DTranspose(
                    filters=f,
                    kernel_size=k,
                    strides=s,
                    padding=p,
                    activation=self.architecture.transfer_fct)(
                    activation)  # shape = (k_samples*batch, height, width, filter)
                print(activation.shape)

            ###############
            # Mean output #
            ###############

            mean_activation = tf.layers.Conv2DTranspose(filters=1,
                                                   kernel_size=k,
                                                   strides=s,
                                                   padding=p,
                                                   )(activation)

            print((self.k_samples, self.batch_size,
                   int(activation.shape[1]), int(activation.shape[2]), 1))

            mean_activation = tf.reshape(mean_activation, (self.k_samples, self.batch_size,
                                                 int(mean_activation.shape[1]), int(mean_activation.shape[2]), 1))
            print("mean_activation:", mean_activation.shape)
            # activation = tf.Print(activation, [tf.reduce_max(activation),tf.reduce_min(activation)])

            mean_activation = tf.nn.sigmoid(mean_activation)
            # activation = tf.Print(activation, [tf.reduce_max(activation),tf.reduce_min(activation)])

            self.x_reconstr_means = tf.reshape(mean_activation, (self.k_samples, self.batch_size, -1))
            print("mean_activation:", mean_activation.shape)

            ################
            # Sigma output #
            ################

            sigma_activation = tf.layers.Conv2DTranspose(filters=1,
                                                   kernel_size=k,
                                                   strides=s,
                                                   padding=p,
                                                   )(activation)
            print("sigma_activation:", sigma_activation.shape)

            sigma_activation = tf.reshape(sigma_activation, (self.k_samples, self.batch_size,
                                                 int(sigma_activation.shape[1]), int(sigma_activation.shape[2]), 1))

            print("sigma_activation:", sigma_activation.shape)

            if self.sigma_param == "1sq":
                x_reconstr_inv_sigma_sq = 1. + tf.square(sigma_activation)
            elif self.sigma_param == "sq":
                x_reconstr_inv_sigma_sq = tf.square(sigma_activation)
            elif self.sigma_param == "exp":
                x_reconstr_inv_sigma_sq = tf.exp(sigma_activation)
            else:
                raise ValueError

            self.x_reconstr_inv_sigma_sq = tf.reshape(x_reconstr_inv_sigma_sq, (self.k_samples, self.batch_size, -1))

            print("sigma_activation:", x_reconstr_inv_sigma_sq.shape)

    def __call__(self, x, average=True):
        """

        :param x: shape = (batch, nvars)
        :param average:
        :return:
        """
        if not self.built:
            self.build(x)
        with tf.variable_scope("activation"):
            z_mu, z_inv_sigma_sq = self.z_mu, self.z_inv_sigma_sq
            zs, lpz = self.zs, self.lpz

            x_expand = tf.expand_dims(x, axis=0)  # shape = (1, batch, n_input)
            # x_expand = tf.expand_dims(x_expand, axis=3)  # shape = (1, batch, n_input, n_filter=1)
            x_tiled = tf.tile(x_expand, multiples=(self.k_samples, 1, 1))  # shape = (k_samples, batch, n_input)

            # log p(x|z)

            x_reconstr_inv_sigma_sq = self.x_reconstr_inv_sigma_sq

            lpxgivenz = tf.check_numerics(
                tf.reduce_sum(gaussian_logprob2(x_tiled, self.x_reconstr_means, x_reconstr_inv_sigma_sq), axis=2),
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


class IWAEconvTFLayer(TFLayer):
    """Constructs K IWAE in a Leaf Layer"""

    def __init__(self, K, scope_id, archi, IWAETYPE=BernoulliIWAE_conv, k_samples=5, sigma_param="1sq"):
        self.archi = archi
        self.IWAETYPE = IWAETYPE
        self.k_samples = k_samples
        self.sigma_param = sigma_param
        super(IWAEconvTFLayer, self).__init__(K, scope_id)

    def build(self, x):
        """

        :param x: a patch of the entire image
        :return:
        """


        ### assemble layer of iwae with possible sharing of inference networks and zs ###

        self.iwaes = []
        for i in range(self.K):
            with tf.variable_scope(str(i)):  # which node in leaflayer
                self.iwaes.append(self.IWAETYPE(architecture=self.archi,
                                                k_samples=self.k_samples,
                                                sigma_param=self.sigma_param
                                                ))

        self.built = True

    def __call__(self, x, average=True):
        """
        :param x: a patch of the image
        :param average: Should the averaging of the importance weighted samples
                        be done at the IWAE leafs or at the root of the SPN
        :return:
        """
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build(x)
            activation = []
            for i, iwae in enumerate(self.iwaes):
                with tf.variable_scope(str(i)):
                    activation.append(iwae(x, average=average))
            if average:
                activation = tf.stack(activation, axis=1)  # shape == (batch, K)
                return activation
            else:
                activation = tf.stack(activation, axis=2)  # shape == (k_samples, batch, K)
                return activation

class IWAEconvTFLayer_Factory(object):
    def __init__(self, x, archi, IWAE_constructor=BernoulliIWAE_conv, k_samples=5, sigma_param="1sq"):
        self.archi = archi
        self.k_samples = k_samples
        self.IWAE_constructor = IWAE_constructor
        self.sigma_param =  sigma_param

        self.leaflayers = []


    def __call__(self, K, scope_id):
        leaflayer = IWAEconvTFLayer(K, scope_id, self.archi, self.IWAE_constructor, sigma_param="1sq",
                           k_samples=self.k_samples)

        self.leaflayers.append(leaflayer)

        return leaflayer