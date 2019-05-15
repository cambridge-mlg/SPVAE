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
Functions to calculate log likelihoods of simple distributions
"""

import tensorflow as tf
import numpy as np

# share the noise constant tensor so that computation graph
# do not have tons of repeated constant nodes
# does this make computation in distributed environment harder?

# noise = tf.constant(1e-7, name="noise")
noise = 1e-7

def gaussian_logprob(x, mu, sigma_sq):
    """
    gaussian distribution parameterised by variance
    :param x: input, a `Tensor` with shape == (batch, 1)
    :param mu: mean parameter, a `Tensor` with shape == (K, )
    :param sigma_sq: variance parameter, a `Tensor` with shape == (K, )
    :return: A `Tensor` with shape == (batch, K)
    :raises:
    """

    diff = tf.subtract(x, mu)  # shape == (batch, K)
    log_exponent = -tf.divide(tf.multiply(diff, diff), 2 * sigma_sq)  # shape == (batch, K)
    log_normalisation = -0.5 * tf.log(noise + 2 * np.pi * sigma_sq)  # shape == (K, )

    activation = log_normalisation + log_exponent  # shape == (batch, K, )
    return tf.check_numerics(activation, "gaussian_logprob")


def gaussian_logprob2(x, mu, inv_sigma_sq):
    """
    gaussian distribution parameterised by precision
    :param x: input, a `Tensor` with shape == (batch, 1)
    :param mu: mean parameter, a `Tensor` with shape == (K, )
    :param inv_sigma_sq: precision parameter, a `Tensor` with shape == (K, )
    :return: a `Tensor` with shape == (batch, K)
    """

    normalisation = inv_sigma_sq / (2 * np.pi)
    with tf.control_dependencies([tf.assert_greater(normalisation, noise)]):
        pass
    diff = tf.subtract(x, mu)  # shape == (batch, K)
    log_exponent = -tf.multiply(tf.multiply(diff, diff), inv_sigma_sq / 2.)  # shape == (batch, K)
    log_normalisation = 0.5 * tf.log(normalisation)  # shape == (K, )

    activation = log_normalisation + log_exponent  # shape == (batch, K, )
    return tf.check_numerics(activation, "gaussian_logprob2")


def standardnormal_logprob(x):
    """

    :param x: input, a `Tensor` with shape == (batch, 1)
    :return: a `Tensor` with shape == (batch, K)
    """
    log_exponent = -0.5 * tf.multiply(x, x)  # shape == (batch, K)
    log_normalisation = -0.5 * tf.log(2 * np.pi)  # shape == (1, )

    activation = log_normalisation + log_exponent  # shape == (batch, K, )
    return activation


def fullcovgaussian_logprob(x, mu, precision):
    """
    multivariate gaussian distribution parameterised by precision
    :param x: input, a `Tensor` with shape == (batch, num_var)
    :param mu: a `Tensor` with shape == (num_var, )
    :param precision: a `Tensor` with  shape == (num_var, num_var); symmetric; positive definite
    :return: a `Tensor` with shape == (batch)
    """
    diff = tf.subtract(x, mu)  # shape == (batch, num_var)
    part1 = tf.matmul(diff, precision)
    variable_dim = len(diff.shape) - 1
    log_exponent = -0.5 * tf.reduce_sum(tf.multiply(diff, part1), axis=variable_dim)  # shape == (batch,)
    log_normalisation = 0.5 * tf.linalg.logdet(1 / (2 * np.pi) * precision)  # shape == (1)
    activation = log_normalisation + log_exponent

    return activation


def bernoulli_logprob(x, rho):
    """

    :param x: binary input, A `Tensor` with shape == (batch, K)
    :param rho: mean parameter between 0 and 1, A `Tensor` with shape == (K)
    :return: a `Tensor` with shape == (batch)
    :raises:
    """

    activation = x * tf.log(noise + rho) + (1. - x) * tf.log(noise + 1. - rho)  # shape == (batch, K)
    print_op = tf.print("logproib", tf.count_nonzero(tf.greater(activation, 0.)))
    with tf.control_dependencies([tf.assert_non_positive(activation, message="logprob error"), tf.check_numerics(activation, "logprob error"), print_op ]):
        return activation


def binomial_logprob(x, rho, N=255):
    """

    :param x:  discrete input in range(0, N), a `Tensor` with shape == (batch, K)
    :param rho: mean parameter between 0 and 1, a `Tensor` with shape == (K)
    :param N: number of categories, integer
    :return: a `Tensor` with shape == (batch)
    """
    # copied from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/distributions/python/ops/binomial.py
    rho = rho * (1. - 2 * noise) + noise
    unnormalized_activation = x * tf.log(noise + rho) + (N - x) * tf.log(noise + 1 - rho)  # shape == (batch, K)
    log_normalization = tf.lgamma(1. + N - x) + tf.lgamma(1. + x) - tf.lgamma(1. + N)
    out = unnormalized_activation - log_normalization
    # out = tf.Print(out, [tf.reduce_max(out)])
    with tf.control_dependencies([tf.assert_non_positive(out)]):
        return tf.check_numerics(out, "binomial_logprob")
    # return out

def concrete_logprob(log_p, log_alpha, temp, axis=2):
    """

    :param log_p: shape == (k_samples, batch, nclasses, K)
    :param log_alpha: shape == (k_samples, batch, nclasses, K)
    :param temp: scalar  shape == ()  or shape == (k_samples, batch, 1, K)  samplewise temperature
    :return:  shape == (k_samples, batch, K)
    """

    log_p = tf.check_numerics(log_p, "log_p")
    log_alpha = tf.check_numerics(log_alpha, "log_alpha")

    n_classes = float(int(log_p.shape[axis]))
    part1 = tf.lgamma(n_classes) + (n_classes - 1) * tf.log(temp)  # shape == (k_samples, batch, 1, K)
    part2 = tf.expand_dims(tf.reduce_sum(log_alpha - (temp + 1) * log_p, axis=axis), axis=axis)  # shape == (k_samples, batch,1, K)
    part3 = -n_classes * tf.expand_dims(tf.reduce_logsumexp(log_alpha - temp * log_p, axis=axis), axis=axis)  # shape == (k_samples, batch, 1, K)

    part1 = tf.check_numerics(part1, "concrete logprob part1")
    part2 = tf.check_numerics(part2, "concrete logprob part2")
    part3 = tf.check_numerics(part3, "concrete logprob part3")

    ans = part1 + part2 + part3
    return tf.check_numerics(ans, "concrete logprob")

def concrete_sample(shape, log_alpha, inv_temp, log_p=False, axis=2):
    """

    :param shape:   (k_samples, batch, nclasses, K)
    :param log_alpha: shape == (k_samples, batch, nclasses, K)
    :param inv_temp: scale   shape == (k_samples, batch, 1, K)
    :param log_p: boolean
    :param axis:
    :return:  shape == (k_samples, batch, nclasses, K)
    """
    G = tf.check_numerics(tf.random_uniform(shape=shape), "Gumbel1")
    G = tf.check_numerics(-tf.log(1e-7 + G), "Gumbel2")
    G = tf.check_numerics(-tf.log(1e-7 + G), "Gumbel3")

    if log_p:
        numerator = (log_alpha + G) * inv_temp  # shape == (k_samples, batch, nclasses, K)
        denominator = tf.expand_dims(tf.reduce_logsumexp(numerator, axis=axis), axis=axis)  # shape == (k_samples, batch, 1, K)
        ans = numerator - denominator
        return tf.check_numerics(ans, "concrete_sample")
    else:
        numerator = tf.exp((log_alpha + G) * inv_temp)   # shape == (k_samples, batch, var)
        denominator = tf.expand_dims(tf.reduce_sum(numerator, axis= axis), axis=axis)   # shape == (k_samples, batch, 1)

        return tf.divide(numerator, denominator)


def KL_diagonalgaussians(mu1, inv_sigma_sq1, mu2, inv_sigma_sq2):
    term1 = 0.5 * (-tf.log(inv_sigma_sq2) + tf.log(inv_sigma_sq1))
    term2 = 0.5 * inv_sigma_sq2  / inv_sigma_sq1
    term3 = 0.5 * tf.square(mu1-mu2) * inv_sigma_sq2
    term4 = -0.5

    return term1 + term2 + term3 + term4


def gaussian_sample(shape, mu, inv_sigma_sq):
    """

    :param shape:
    :param mu:
    :param inv_sigma_sq:
    :return:
    """
    z = tf.random_normal(shape, dtype=tf.float32)
    zs = tf.divide(z, tf.sqrt(inv_sigma_sq)) + mu
    return zs

def gaussian_sample2(shape, mu, sigma):
    """

    :param shape:
    :param mu:
    :param sigma:
    :return:
    """
    z = tf.random_normal(shape, dtype=tf.float32)
    zs = tf.multiply(z, sigma) + mu
    return zs


def bernoulli_sample(shape, rho):
    """

    :param shape:
    :param rho:
    :return:
    """
    pass

def binomial_sample(shape, rho):
    """

    :param shape:
    :param rho:
    :return:
    """
