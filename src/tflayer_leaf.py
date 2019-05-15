# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
SPN Leaf implemented as TFLayers
"""

from src.tfdensitynet import *
from src.tfdistributions import *

"""
I designed TFLayer API to follow Keras functional API.
This will allow easy layer reuse

All leaf TFLayers are constructed with signature:

(K, scope_id, debug=False)
:param K: shape = number of distributions per scope
:param scope_id: an integer for naming portions of computation graph
:param debug: set weights to non random values

All leaf TFLayers are called with signature:
(x)
:param x: input `Tensor`
:return `Tensor`

Basic assertion checks:
* All final activation computed have len(activation.shape) == 2

"""


class TFLayer(object):
    def __init__(self, K, scope_id):
        self.K = K
        self.scope_id = scope_id
        self.built = False


#######################
### Analytic leaves ###
#######################

###################
# Gaussian leaves #
###################

class GaussianTFLayer(TFLayer):
    def __init__(self, K, scope_id, debug=False):
        self.debug = debug
        super(GaussianTFLayer, self).__init__(K, scope_id)

    def build(self):
        if self.debug:
            self._mu = tf.constant(0,
                                   name="mu",
                                   dtype=tf.float32,
                                   shape=(self.K,))
            self._inv_sigma_sq = tf.constant(1,
                                             name="inv_sigma_sq",
                                             dtype=tf.float32,
                                             shape=(self.K,))
        else:
            self._mu = tf.get_variable(name="mu",
                                       dtype=tf.float32,
                                       shape=(self.K,),
                                       initializer=tf.initializers.random_normal())  # shape == (K,)
            _inv_sigma_sq = tf.get_variable(name="inv_sigma_sq",
                                            dtype=tf.float32,
                                            shape=(self.K,),
                                            initializer=tf.initializers.random_normal())  # shape == (K,)
            # ensure positive non-zero sigma_sq
            self._inv_sigma_sq = noise + tf.square(_inv_sigma_sq)  # shape == (K,)

        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build()

            assert int(x.shape[1]) == 1

            X = tf.tile(x, multiples=[1, self.K])  # shape == (batch, K)
            activation = gaussian_logprob2(X, self._mu, self._inv_sigma_sq)  # shape == (batch, K, )

            assert len(activation.shape) == 2
            return activation


class MultivariateDiagGaussianTFLayer(TFLayer):
    def __init__(self, K, scope_id, debug=False):
        self.debug = debug
        super(MultivariateDiagGaussianTFLayer, self).__init__(K, scope_id)

    def build(self, x):
        self.num_variables = int(x.shape[1])
        if self.debug:
            self._mu = tf.constant(0,
                                   name="mu",
                                   dtype=tf.float32,
                                   shape=(self.K, self.num_variables))
            self._inv_sigma_sq = tf.constant(1, name="sigma_sq", dtype=tf.float32,
                                             shape=(self.K, self.num_variables))
        else:
            self._mu = tf.get_variable(name="mu",
                                       dtype=tf.float32,
                                       shape=(self.K, self.num_variables),
                                       initializer=tf.initializers.random_normal())  # shape == (K, num_var)
            _inv_sigma = tf.get_variable(name="inv_sigma",
                                         dtype=tf.float32,
                                         shape=(self.K, self.num_variables),
                                         initializer=tf.initializers.random_normal())  # shape == (K, num_var)
            # ensure positive non-zero sigma_sq
            self._inv_sigma_sq = noise + tf.square(_inv_sigma)  # shape == (K, num_var)
        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build(x)

            assert int(x.shape[1]) == int(self._mu.shape[1])

            x_expanded = tf.expand_dims(x, axis=1)  # shape == (batch, 1, num_variables)
            X = tf.tile(x_expanded, multiples=[1, self.K, 1])  # shape == (batch, K, num_variables)
            activation = tf.reduce_sum(gaussian_logprob2(X, self._mu, self._inv_sigma_sq),
                                       axis=2)  # shape == (batch, K, )

            assert len(activation.shape) == 2  # shape == (batch, K)
            return activation


class MultivariateFullCovGaussianTFLayer(TFLayer):
    def __init__(self, K, scope_id, debug=False):
        self.debug = debug
        super(MultivariateFullCovGaussianTFLayer, self).__init__(K, scope_id)

    def build(self, x):
        self.num_variables = int(x.shape[1])
        if self.debug:
            _mu = tf.constant(0, name="mu", dtype=tf.float32, shape=(self.K, self.num_variables))
            _precision = tf.eye(self.num_variables, dtype=tf.float32, name="precision")
            self.mu = _mu
            self.precision = _precision
        else:
            self._mu = tf.get_variable(name="mu",
                                       dtype=tf.float32,
                                       shape=(self.K, self.num_variables),
                                       initializer=tf.initializers.random_normal())  # shape == (K, num_var)
            _sigma_inv = tf.get_variable(name="sigma",
                                         dtype=tf.float32,
                                         shape=(self.K, self.num_variables, self.num_variables),
                                         initializer=tf.initializers.random_normal())  # shape == (K, num_var, num_var)
            _sigma_inv_sq = tf.square(_sigma_inv)  # shape == (K, num_var, num_var)
            # TODO: think about how to protect the precision matrix
            self._precision = _sigma_inv_sq + tf.transpose(_sigma_inv_sq,
                                                           perm=[0, 2, 1])  # shape == (K, num_var, num_var)
        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build(x)

            assert int(x.shape[1]) == int(self._mu.shape[1])

            x_expanded = tf.expand_dims(x, axis=1)  # shape == (batch, 1, num_variables)
            X = tf.tile(x_expanded, multiples=[1, self.K, 1])  # shape == (batch, K, num_variables)

            activation = fullcovgaussian_logprob(X, self._mu, self._precision)

            assert len(activation.shape) == 2  # shape == (batch, K)
            return activation


####################
# Bernoulli leaves #
####################

class BernoulliTFLayer(TFLayer):
    def __init__(self, K, scope_id, debug=False):
        self.debug = debug
        super(BernoulliTFLayer, self).__init__(K, scope_id)

    def build(self):
        if self.debug:
            probs = tf.constant(0.5, shape=(self.K,), dtype=tf.float32)
            self.probs = probs
        else:
            weights = tf.get_variable(name="weights",
                                      dtype=tf.float32,
                                      shape=(self.K,),
                                      initializer=tf.initializers.random_normal())  # shape == (K,)
            self.probs = tf.nn.sigmoid(weights, name="probs")  # shape == (K,)
        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build()

            assert int(x.shape[1]) == 1

            X = tf.tile(x, multiples=[1, self.K])  # shape == (batch, K)
            activation = bernoulli_logprob(X, self.probs)  # shape == (batch, K)

            assert len(activation.shape) == 2  # shape == (batch, K)
            return activation


class MultivariateBernoulliTFLayer(TFLayer):
    def __init__(self, K, scope_id, debug=False):
        self.debug = debug
        super(MultivariateBernoulliTFLayer, self).__init__(K, scope_id)

    def build(self, x):
        self.num_variables = int(x.shape[1])
        if self.debug:
            probs = tf.constant(0.5, shape=(self.K, self.num_variables), dtype=tf.float32)
            self.probs = probs
        else:
            weights = tf.get_variable(name="weights",
                                      dtype=tf.float32,
                                      shape=(self.K, self.num_variables),
                                      initializer=tf.initializers.random_normal())  # shape == (K, num_var)
            self.probs = tf.nn.sigmoid(weights, name="probs")  # shape == (K, num_var)
        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build(x)

            assert int(x.shape[1]) == int(self.probs.shape[1])

            x_expanded = tf.expand_dims(x, axis=1)  # shape == (batch, 1, num_var)
            X = tf.tile(x_expanded, multiples=[1, self.K, 1])  # shape == (batch, K, num_var)
            activation = bernoulli_logprob(X, self.probs)  # shape == (batch, K, num_var)
            activation = tf.reduce_sum(activation, axis=2)

            assert len(activation.shape) == 2
            return activation


###################
# Binomial leaves #
###################

class BinomialTFLayer(TFLayer):
    def __init__(self, K, scope_id, debug=False):
        self.debug = debug
        super(BinomialTFLayer, self).__init__(K, scope_id)

    def build(self):
        if self.debug:
            probs = tf.constant(0.5, shape=(self.K,), dtype=tf.float32)
            self.probs = probs
        else:
            weights = tf.get_variable(name="weights",
                                      dtype=tf.float32,
                                      shape=(self.K,),
                                      initializer=tf.initializers.random_normal())  # shape == (K,)
            self.probs = tf.nn.sigmoid(weights, name="probabilities")  # shape == (K,)
        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build()

            assert int(x.shape[1]) == 1

            X = 255 * tf.tile(x, multiples=[1, self.K])  # shape == (batch, K)
            activation = binomial_logprob(X, self.probs, 255)  # shape == (batch, K)

            assert len(activation.shape) == 2
            return activation


class MultivariateBinomialTFLayer(TFLayer):
    def __init__(self, K, scope_id, debug=False):
        self.debug = debug
        super(MultivariateBinomialTFLayer, self).__init__(K, scope_id)

    def build(self, x):
        self.num_variables = int(x.shape[1])
        if self.debug:
            probs = tf.constant(0.5, shape=(self.K, self.num_variables), dtype=tf.float32)
            self.probs = probs
        else:
            weights = tf.get_variable(name="weights",
                                      dtype=tf.float32,
                                      shape=(self.K, self.num_variables),
                                      initializer=tf.initializers.random_normal())  # shape == (K,num_var)
            self.probs = tf.nn.sigmoid(weights, name="probabilities")  # shape == (K,num_var)
        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build(x)

            assert int(x.shape[1]) == int(self.probs.shape[1])

            x_expanded = tf.expand_dims(x, axis=1)  # shape == (batch, 1, num_var)
            X = 255 * tf.tile(x_expanded, multiples=[1, self.K, 1])  # shape == (batch, K, num_var)
            activation = binomial_logprob(X, self.probs, 255)  # shape == (batch, K, num_var)
            activation = tf.reduce_sum(activation, axis=2)  # shape == (batch, K,)

            assert len(activation.shape) == 2
            return activation


######################################
### Wrapping DenseNet into TFLayer ###
######################################

class DenseNetTFLayer(TFLayer):
    def __init__(self, K, scope_id, architecture, densenet_constructor=BernoulliDenseNet, k_samples=50,
                 nz=1, shared_z="none", provided_zs=None):
        """Constructs K Density Networks within a leaf layer"""

        self.architecture = architecture
        self.densenet_constructor = densenet_constructor
        self.k_samples = k_samples
        assert isinstance(architecture, DenseNetArchitecture)

        self.nz = nz
        self.shared_z = shared_z
        self.global_zs = provided_zs
        super(DenseNetTFLayer, self).__init__(K, scope_id)

    def build(self, x):
        # four ways that zs can be shared
        if self.shared_z == "global":
            assert self.global_zs is not None
            zs = [self.global_zs(x)] * self.K
        elif self.shared_z == "layer":
            zs = tf.random_normal(shape=(self.k_samples, tf.shape(x)[0], self.nz), dtype=tf.float32)
            zs = [zs] * self.K
        elif self.shared_z == "none":
            zs = [None] * self.K
        elif self.shared_z == "across":
            zs = [self.global_zs(i)(x) for i in range(self.K)]
        else:
            raise ValueError

        assert type(zs) is list

        # creation
        self.densenets = []
        for i in range(self.K):
            with tf.variable_scope(str(i)):  # which node in leaflayer
                self.densenets.append(self.densenet_constructor(architecture=self.architecture,
                                                                k_samples=self.k_samples,
                                                                nz=self.nz,
                                                                zs=zs[i]))
        self.built = True

    def __call__(self, x):
        with tf.variable_scope("Leaf" + str(self.scope_id)):
            if not self.built:
                self.build(x)

            activation = []
            for i, densenet in enumerate(self.densenets):
                with tf.variable_scope(str(i)):
                    activation.append(densenet(x))
            activation = tf.stack(activation, axis=1)  # shape == (batch, K)
            return activation


class DenseNetTFLayer_Factory(object):
    def __init__(self, architecture, densenet_constructor, k_samples=50, nz=1, shared_zs="none"):
        """
        A wrapper around DenseNetTFLayer to help it comply with (K, scope_id) signature of other TFLayers


        :param architecture:
        :param densenet_constructor: Single Density Network as leaf node
        :param k_samples: number of samples to use for monte carlo integration
        :param nz: dimension of latent variable.
        :param shared_zs: str, specifies how the latent variable z is shared within SP-DensityNet
            either "global", "layer" or "none"
        """
        self.archi = architecture
        self.densenet = densenet_constructor
        self.k_samples = k_samples
        self.nz = nz
        self.shared_z = shared_zs

        if shared_zs == "global":
            self.global_zs = RandomNormal(self.k_samples, self.nz)
        elif shared_zs == "layer" or shared_zs == "none":
            self.global_zs = None
        elif shared_zs == "across":
            self.global_zs = ManyRandomNormal(self.k_samples, self.nz)
        else:
            raise ValueError()

    def __call__(self, K, scope_id):
        return DenseNetTFLayer(K, scope_id, self.archi, self.densenet, self.k_samples,
                               nz=self.nz, shared_z=self.shared_z, provided_zs=self.global_zs)


class ManyRandomNormal(object):
    def __init__(self, k_samples, nz):
        self.k_samples = k_samples
        self.nz = nz
        self.rn = {}

    def __call__(self, i):
        if i not in self.rn:
            self.rn[i] = RandomNormal(self.k_samples, self.nz)
        return self.rn[i]


##################################
### Wrapping IWAE into TFLayer ###
##################################

class IWAETFLayer(TFLayer):
    """Constructs K IWAE in a Leaf Layer"""

    def __init__(self, K, scope_id, recog_archi, gener_archi, IWAETYPE=BernoulliIWAE, k_samples=5):
        """

        :param K: int
            number of leaf nodes in the layer
        :param scope_id: int
            unique id of the scope

        # the following at VAE parameters
        :param recog_archi:
        :param gener_archi:
        :param IWAETYPE:
        :param k_samples:

        """
        self.recog_archi = recog_archi
        self.gener_archi = gener_archi
        self.IWAETYPE = IWAETYPE
        self.k_samples = k_samples

        super(IWAETFLayer, self).__init__(K, scope_id)

    def build(self, x):
        """

        :param x: a patch of the entire image
        :return:
        """

        ### assemble layer of iwae with possible sharing of inference networks and zs ###

        self.iwaes = []
        for i in range(self.K):
            with tf.variable_scope(str(i)):  # which node in leaflayer
                self.iwaes.append(self.IWAETYPE(recog_architecture=self.recog_archi,
                                                gener_architecture=self.gener_archi,
                                                k_samples=self.k_samples,
                                                qz_params=None
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

    @property
    def get_zs(self):
        if not hasattr(self, "_zs"):
            self._zs = tf.concat([zs for (z_mu, z_inv_sigma_sq, zs) in self.qz_params], axis=1)
        return self._zs


class ManyInferenceNet(object):
    def __init__(self, recog_archi, k_samples, x):
        self.recog_archi = recog_archi
        self.k_samples = k_samples
        self.x = x
        self.inference_nets = {}

    def __call__(self, key):
        """

        :param key: a unique id associated with an inference network.
            choice of key is left to layer_factory to allow for appropriate sharing of inference network
        :return: qz_params == (z_mu, z_inv_sigma_sq, zs)
        """
        if key not in self.inference_nets:
            qz_params = InferenceNet(self.recog_archi, self.k_samples)(self.x)
            self.inference_nets[key] = qz_params
        return self.inference_nets[key]


class IWAETFLayer_Factory_globalinference(object):
    def __init__(self, x, recog_archi, gener_archi, IWAE_constructor=BernoulliIWAE, k_samples=5,
                 share_latent="none", share_inferencenet="none"):
        """

        :param x:
        :param recog_archi:
        :param gener_archi:
        :param IWAE_constructor:
        :param k_samples:
        """
        self.recog_archi = recog_archi
        self.gener_archi = gener_archi
        self.k_samples = k_samples
        self.IWAE_constructor = IWAE_constructor

        self.leaflayers = []


    def __call__(self, K, scope_id):
        leaflayer = IWAETFLayer(K, scope_id, self.recog_archi, self.gener_archi, self.IWAE_constructor, k_samples=self.k_samples)

        self.leaflayers.append(leaflayer)

        return leaflayer