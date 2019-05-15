import tensorflow as tf
from tflayer import TFLayer
from src.tflayer_leaf import BernoulliTFLayer
import logging

"""
Tensorial Decomposition implementation of Poon Domingos SPN
"""


class ScopeTFLayer(TFLayer):
    def __init__(self, Ltensor, Rtensor, K, scope_id):
        """

        :param Ltensor: shape == (batch, rho, K1)
        :param Rtensor: shape == (batch, rho, K2)
        :param K:
        :param scope_id:
        """

        assert(len(Ltensor.shape)==3)
        assert(len(Rtensor.shape)==3)

        K1 = int(Ltensor.shape[2])
        K2 = int(Rtensor.shape[2])

        rho = int(Ltensor.shape[1])

        assert rho == int(Rtensor.shape[1])

        with tf.variable_scope("Scope"+str(scope_id)):
            weightsA = tf.get_variable(name="A", shape=(K,rho), dtype=tf.float32,
                                         initializer=tf.initializers.random_normal(),
                                         regularizer=tf.contrib.layers.l1_regularizer(0.1))  # shape = (K, rho)

            weightsB = tf.get_variable(name="B", shape=(rho, K1*K2), dtype=tf.float32,
                                         initializer=tf.initializers.random_normal(),
                                         regularizer=tf.contrib.layers.l1_regularizer(0.1))  # shape = (rho, K^2)

            logA = tf.nn.log_softmax(weightsA, axis=1)  # shape = (K, rho)
            logB = tf.reshape(tf.nn.log_softmax(weightsB, axis=1), shape=(rho, K1,K2))  # shape = (rho, K1, K2)

            Ltensor = tf.expand_dims(Ltensor, axis=3)  # shape = (batch, rho, K1, 1)
            Rtensor = tf.expand_dims(Rtensor, axis=2)  # shape = (batch, rho, 1, K2)

            tensor = Ltensor + Rtensor  # shape = (batch, rho, K1, K2)
            # print("1", tensor.shape)
            tensor = logB + tensor  # shape = (batch, rho, K1, K2)
            # print("2",tensor.shape)
            tensor = tf.reduce_logsumexp(tf.reduce_logsumexp(tensor, axis=3), axis=2)  # shape = (batch, rho)
            # print("3",tensor.shape)
            tensor = tf.expand_dims(tensor, axis=1)  # shape = (batch, 1, rho)
            # print("4",tensor.shape)
            tensor = tensor + logA  # shape = (batch, K, rho)

            activation = tf.reduce_logsumexp(tensor, axis=2, name="activation")
            super(ScopeTFLayer, self).__init__(activation)

###########################
# Superpixel architecture #
###########################

class SP_SPVAE_Factory(object):
    """
    This class handles the creation mechanism of ScopeTFLayer instances for the superpixel architecture of SPN

    Every ScopeTFLayer uniquely represents K distributions over some set of random variables
    """
    def __init__(self, leaflayer_constructor):
        self._scopetensors = {}  # scopeblock instance id as key
        self.LeafLayer = leaflayer_constructor

    def get_sumlayer_instance(self, scopeblock, K=None):

        # Sum Layer already constructed
        if scopeblock.id in self._scopetensors:
            return self._scopetensors[scopeblock.id]
        # Sum Layer not yet constructed
        else:
            assert K is not None

            LTensor, RTensor = [], []

            # partition blocks are ordered by id
            for partitionblock in sorted(scopeblock.children, key=lambda partitionblock: partitionblock.id):
                # child scopeblocks are ordered by id
                child_scope_pair = sorted(partitionblock.children, key=lambda cs: cs.id)

                LTensor.append(self._scopetensors[child_scope_pair[0].id].activation)
                RTensor.append(self._scopetensors[child_scope_pair[1].id].activation)

            LTensor = tf.stack(LTensor, axis=1)  # shape == (batch, rho, K)
            RTensor = tf.stack(RTensor, axis=1)  # shape == (batch, rho, K)


            # construct sum layer
            obj = ScopeTFLayer(LTensor, RTensor, K, scopeblock.id)
            self._scopetensors[scopeblock.id] = obj
            return obj

    def get_leaflayer_instance(self, x, K, scope_id):
        """Every leaf TFLayer object instantiated has a unique set of random_variables"""

        if scope_id in self._scopetensors:
            return self._scopetensors[scope_id]
        else:
            obj = self.LeafLayer(x, K, scope_id)
            self._scopetensors[scope_id] = obj
            return obj


class SP_SPVAE(object):
    """
    Superpixel architecture of SP-VAE
    """
    def __init__(self, x, region_graph, leaflayer=BernoulliTFLayer, K=3):
        # SPN structure
        self.K = K

        # Problem Property
        self.num_variables = len(region_graph.root.vars)

        # input
        self.x = x

        self.LeafLayer = leaflayer
        self.root, _ = self.parse_region_graph(region_graph)

        self.activation = self.root.activation
        print("activation complete")

    def parse_region_graph(self, region_graph):
        # smart constructor
        tensor_factory = SP_SPVAE_Factory(self.LeafLayer)
        # output variable
        leaves = []

        print("start leaf gather")

        for leaf_scope in region_graph.get_leaf_scopes():
            # slice the data matrix
            x_subset = tf.gather(self.x, sorted(leaf_scope.vars), axis=1)  # shape == (batch, num of variables)
            # create LeafLayer
            leaflayer = tensor_factory.get_leaflayer_instance(x=x_subset, K=self.K, scope_id=leaf_scope.id)
            leaves.append(leaflayer)

        print("start traversal")
        scopes, partitions = region_graph.toposort_traverse_scopes(large2small=False)
        print("end traversal")
        # first list within scopes is a list of leaf scopes.
        # They are already processed. Hence skipped
        scopes = scopes[1:]

        for partitions_with_same_rank, scopes_with_same_rank in zip(partitions, scopes):
            print("level" + " "+ str(len(partitions_with_same_rank)) + " " + str(len(scopes_with_same_rank)))
            logging.debug("level" + " "+ str(len(partitions_with_same_rank)) + " " + str(len(scopes_with_same_rank)))

            for scope in scopes_with_same_rank:
                # testing if root scope
                if len(scope.vars) == self.num_variables:
                    sumlayer = tensor_factory.get_sumlayer_instance(scope, 1)
                else:
                    sumlayer = tensor_factory.get_sumlayer_instance(scope, self.K)

        return sumlayer, leaves  # root, leaves

##############################
# Convolutional architecture #
##############################

class Conv_SPVAE_Factory(object):
    """
    This class handles the creation mechanism of ScopeTFLayer instances for the convolutional architecture of SPN

    Every ScopeTFLayer uniquely represents K distributions over some set of random variables
    """
    def __init__(self, vaeleaflayer_constructor, bgleaflayer_constructor):
        self._scopetensors = {}  # scopeblock instance id as key
        self.VAELeafLayer = vaeleaflayer_constructor
        self.BGLeafLayer = bgleaflayer_constructor

    def get_sumlayer_instance(self, scopeblock, K=None):

        # Sum Layer already constructed
        if scopeblock.id in self._scopetensors:
            return self._scopetensors[scopeblock.id]
        # Sum Layer not yet constructed
        else:
            assert K is not None

            LTensor, RTensor = [], []

            # partition blocks are ordered by id
            for partitionblock in sorted(scopeblock.children, key=lambda partitionblock: partitionblock.id):
                # child scopeblocks are ordered by id
                child_scope_pair = sorted(partitionblock.children, key=lambda cs: cs.id)

                LTensor.append(self._scopetensors[child_scope_pair[0].id].activation)
                RTensor.append(self._scopetensors[child_scope_pair[1].id].activation)

            LTensor = tf.stack(LTensor, axis=1)  # shape == (batch, rho, K)
            RTensor = tf.stack(RTensor, axis=1)  # shape == (batch, rho, K)


            # construct sum layer
            obj = ScopeTFLayer(LTensor, RTensor, K, scopeblock.id)
            self._scopetensors[scopeblock.id] = obj
            return obj

    def get_bgleaflayer_instance(self, x, K, scope_id):
        """Every leaf TFLayer object instantiated has a unique set of random_variables"""

        if scope_id in self._scopetensors:
            return self._scopetensors[scope_id]
        else:
            obj = self.BGLeafLayer(x, K, scope_id)
            self._scopetensors[scope_id] = obj
            return obj

    def get_vaeleaflayer_instance(self, x, K, scope_id):
        """Every leaf TFLayer object instantiated has a unique set of random_variables"""

        if scope_id in self._scopetensors:
            return self._scopetensors[scope_id]
        else:
            obj = self.VAELeafLayer(x, K, scope_id)
            self._scopetensors[scope_id] = obj
            return obj

class Conv_SPVAE(object):
    """Convolutional architecture of SPN"""
    def __init__(self, x, region_graph, vaeleaflayer, bgleaflayer = BernoulliTFLayer, K=3):
        # SPN structure
        self.K = K

        # Problem Property
        self.num_variables = len(region_graph.root.vars)

        # input
        self.x = x

        self.VAELeafLayer = vaeleaflayer
        self.BGLeafLayer = bgleaflayer
        self.root, _ = self.parse_region_graph(region_graph)

        self.activation = self.root.activation
        print("activation complete")

    def parse_region_graph(self, region_graph):
        # smart constructor
        factory = Conv_SPVAE_Factory(self.VAELeafLayer, self.BGLeafLayer)
        # output variable
        leaves = []

        print("start leaf gather")

        for leaf_scope in region_graph.get_leaf_scopes():
            # slice the data matrix
            x_subset = tf.gather(self.x, sorted(leaf_scope.vars), axis=1)  # shape == (batch, num of variables)

            if region_graph.is_smaller_than_vae(leaf_scope):
                # create bgLeafLayer
                leaflayer = factory.get_bgleaflayer_instance(x=x_subset, K=self.K, scope_id=leaf_scope.id)
            else:
                # create VAELeafLayer
                leaflayer = factory.get_vaeleaflayer_instance(x=x_subset, K=self.K, scope_id=leaf_scope.id)

            leaves.append(leaflayer)

        print("start traversal")
        scopes, partitions = region_graph.toposort_traverse_scopes(large2small=False)
        print("end traversal")
        # first list within scopes is a list of leaf scopes.
        # They are already processed. Hence skipped
        scopes = scopes[1:]

        for partitions_with_same_rank, scopes_with_same_rank in zip(partitions, scopes):
            print("level" + " "+ str(len(partitions_with_same_rank)) + " " + str(len(scopes_with_same_rank)))
            logging.debug("level" + " "+ str(len(partitions_with_same_rank)) + " " + str(len(scopes_with_same_rank)))

            for scope in scopes_with_same_rank:
                # testing if root scope
                if len(scope.vars) == self.num_variables:
                    sumlayer = factory.get_sumlayer_instance(scope, 1)
                else:
                    sumlayer = factory.get_sumlayer_instance(scope, self.K)

        return sumlayer, leaves  # root, leaves
