import tensorflow as tf
from src.tfdensitynet import MLP2
from src.tflayer_leaf import TFLayer
import time


class IWSumTFLayer(TFLayer):
    """
    k_samples from IWAE leaves are allowed to pass through the SPN.
    Main difference from SumTFLayer is the shape of the sum weight tensor
    """

    def __init__(self, K, scope_id,):
        """
        param K: number of nodes in SumLayer (integer)
        param scope_id: (integer)
        """
        super(IWSumTFLayer, self).__init__(K, scope_id)

    def build(self, productlayers):
        """
        associate a weight matrix with sumlayer
        """
        # K1 = int(productlayers.shape[1])
        # K2 = int(productlayers.shape[2])
        # numlayers = int(productlayers.shape[3])
        #
        # num_nodes = K1 * K2 * numlayers * self.K

        # shape == (k_samples, batch, K^2)
        nprods = int(productlayers.shape[2])
        k_samples = int(productlayers.shape[0])
        batch_size = tf.shape(productlayers)[1]

        self.W = tf.get_variable(name="weights",
                                 shape=[1, 1, nprods, self.K],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.random_normal(),
                                 regularizer=tf.contrib.layers.l1_regularizer(0.01))  # shape = (1, 1, nprods, K)

        self.log_p = self.W - tf.expand_dims(tf.reduce_logsumexp(self.W, axis=2),
                                             axis=2)  # shape = (k_samples, batch, K^2, K)
        self.built = True

    def __call__(self, productlayers):
        """
        multiply weight matrix with child layer
        :param productlayers: list of `Tensor`s of shape == (k_samples, batch, K, K)
        :return: a `Tensor` of shape == (k_samples, batch, K)
        """
        with tf.variable_scope("SumLayer" + str(self.scope_id)):
            # this contains a copy operation !! slow!!
            assert type(productlayers) == list
            assert len(productlayers) > 0
            productlayers = tf.concat(productlayers, axis=2)  # shape == (k_samples, batch, K^2)

            if not self.built:
                self.build(productlayers)

            # # top pass of activation
            prods = tf.expand_dims(productlayers, axis=3)  # shape = (k_samples, batch, K^2, 1)
            out = prods + self.log_p  # exploit broadcasting # shape = (k_samples, batch, K^2 K)
            activation = tf.reduce_logsumexp(out, axis=2,
                                             name="activation")  # shape == (k_samples, batch, K)

            print(activation.shape)
            return activation



def ProductOp(sumlayer1, sumlayer2, partition_id):
    """
    cartesian product between two layers
    :param sumlayer1: a `Tensor` of shape == (batch, K1)
    :param sumlayer2: a `Tensor` of shape == (batch, K2)
    :param partition_id: integer
    :return: a `Tensor` of shape == (batch, K1*K2)
    """
    with tf.variable_scope("ProdLayer" + str(partition_id)):
        # top pass of activation
        sl1 = tf.expand_dims(sumlayer1, axis=2)  # shape = (k_samples, batchsize, 1, K)
        sl2 = tf.expand_dims(sumlayer2, axis=3)  # shape = (k_samples, batchsize, K, 1)

        # catesian product in log domain
        # exploit broadcasting
        batch_outer = tf.add(sl1, sl2, name = "activation")  # shape = (k_samples, batchsize, K, K)
        # return batch_outer

        # need to flatten tensor.
        # Otherwise we cannot concatenate it with
        # another productlayer of different K
        k_samples = int(sumlayer1.shape[0])
        batch_size = tf.shape(sumlayer1)[1]
        n_nodes1 = int(sumlayer1.shape[2])
        n_nodes2 = int(sumlayer2.shape[2])

        activation = tf.reshape(batch_outer, [k_samples, batch_size, n_nodes1*n_nodes2],
                                name="activation")  # shape = (k_samples, batchsize, K^2)

        return activation


class IWSP_TFLayerFactory(object):
    """
    This class encapsulates the creation and storage of SumTFLayer and ProductTFLayer.
    Every SumTFLayer represents K distributions over some set of random variables
    Every ProductTFLayer represents K^2 distributions over some set of random variables

    Block Graph handles the indexing and relationships between ScopeBlocks and PartitionBlocks

    ScopeBlocks and PartitionBlocks are thus used as construction arguments

    There is one way to index the SumTFLayers:
        1) scope block instance id

    There is one way to index the ProductTFLayers:
        1) partition block instance id

    These two methods were chosen to minimize tensorflow's op name overhead in the computation graph
    """

    def __init__(self, leaflayer_constructor):
        self.LeafLayer = leaflayer_constructor

        # activations
        self._sumlayers_activation = {}  # scopeblock instance id as key
        self._productlayers_activation = {}  # partitionblock instance id as key

        # model
        self.leaflayer = {}
        self.sumlayer = {}

        # reconstructions
        self.sumlayer_reconstruction = {}
        self.productlayer_reconstruction = {}


    def get_iwsumlayer_activation(self, scopeblock, K=None):
        """
        get sumlayer. evaluate if not yet computed
        param scopeblock: object describing the scope of the sumlayer and its parents and children
        param K: integer, number of nodes in sumlayer
        return: a `Tensor` of shape == (batch, K)
        """
        # Sum Layer already constructed
        if scopeblock.id in self._sumlayers_activation:
            return self._sumlayers_activation[scopeblock.id]
        # Sum Layer not yet constructed
        else:
            assert K is not None
            # collect product layers into a sorted list
            child_productlayers = [self.get_productlayer_activation(partitionblock)
                                   for partitionblock in sorted(scopeblock.children, key=lambda x: x.id)]

            # evaluate sum layer
            sl = IWSumTFLayer(K, scopeblock.id)
            self.sumlayer[scopeblock] = sl
            sumlayer = sl(child_productlayers)
            self._sumlayers_activation[scopeblock.id] = sumlayer
            return sumlayer

    def get_leaflayer_activation(self, x, K, scope_id):
        """
        get leaflayer. evaluate if not yet computed
        param x: input data, `Tensor` of shape == (batch, num_var)
        param K: number of nodes in leaflayer, integer
        param scope_id: integer, Every leaflayer has a unique scope
        return: a `Tensor` of shape == (batch, K)
        """

        if scope_id in self._sumlayers_activation:
            return self._sumlayers_activation[scope_id]
        else:
            leaf = self.LeafLayer(K, scope_id)
            self.leaflayer[scope_id] = leaf
            leaflayer = leaf(x, average=False)
            self._sumlayers_activation[scope_id] = leaflayer
            return leaflayer

    def get_productlayer_activation(self, partitionblock):
        """
        get productlayer. evaluate if not yet computed
        param partitionblock: object describing the scope of the productlayer, its parents and children
        return: a `Tensor` of shape == (batch, K^2)
        """

        if partitionblock.id in self._productlayers_activation:
            return self._productlayers_activation[partitionblock.id]
        else:
            assert len(partitionblock.children) == 2
            scopeblock1, scopeblock2 = partitionblock.children

            # sumlayer1 = self.get_variationalsumlayer_activation(scopeblock1)
            # sumlayer2 = self.get_variationalsumlayer_activation(scopeblock2)
            sumlayer1 = self.get_iwsumlayer_activation(scopeblock1)
            sumlayer2 = self.get_iwsumlayer_activation(scopeblock2)

            prodlayer = ProductOp(sumlayer1, sumlayer2, partitionblock.id)
            self._productlayers_activation[partitionblock.id] = prodlayer
            return prodlayer


class IWSPNTFLayer(TFLayer):
    """
    Usage:
    # SPN is a function built recursively from simpler functions
    px = SPNTFLayer(region_graph, BernoulliTFLayer, 3, 3)(x)
    """

    def __init__(self, K, scope_id, region_graph, leaflayer, sumK=3, leafK=3):
        """
        :param K:
        :param scope_id:
        :param region_graph: architecture / structure of SPN
        :param leaflayer: distribution type of SPN
        :param sumK: number of nodes per sumlayer
        :param leafK: number of nodes per leaflayer
        """

        # SPN structure
        self.sumK = sumK
        self.leafK = leafK
        self.region_graph = region_graph
        self.num_variables = len(region_graph.root.vars)

        # SPN distribution type
        self.LeafLayer = leaflayer
        self.splayer_factory = IWSP_TFLayerFactory(self.LeafLayer)
        super(IWSPNTFLayer, self).__init__(K, scope_id)

    def __call__(self, x):
        """
        param x: input `Tensor` of shape == (batch, num_var)
        return: a `Tensor` of shape == (batch, 1)
        """
        NUM_LEAF_NODES = 0
        NUM_SUM_NODES = 0
        NUM_PROD_NODES = 0
        print("start leaf gather")

        for leaf_scope in self.region_graph.get_leaf_scopes():
            # pick columns from the data matrix
            x_subset = tf.gather(x, sorted(leaf_scope.vars), axis=1)  # shape == (batch, num of variables)
            # create LeafLayer
            leaflayer = self.splayer_factory.get_leaflayer_activation(x=x_subset, K=self.leafK, scope_id=leaf_scope.id)  # (k_samples, batch, K)
            NUM_LEAF_NODES += int(leaflayer.shape[2])

        print("start traversal")
        scopes, partitions = self.region_graph.toposort_traverse_scopes(large2small=False)
        print("end traversal")

        start_time = time.time()
        # first list within scopes is a list of leaf scopes.
        # They are already processed. Hence skipped
        scopes = scopes[1:]

        for partitions_with_same_rank, scopes_with_same_rank in zip(partitions, scopes):
            print(str(len(partitions_with_same_rank)) + " partitions, " + str(len(scopes_with_same_rank)) + " scopes")

            for partition in partitions_with_same_rank:
                productlayer = self.splayer_factory.get_productlayer_activation(partition)  # (k_samples, batch, K)
                NUM_PROD_NODES += int(productlayer.shape[2])

            for scope in scopes_with_same_rank:
                # testing if root scope
                if len(scope.vars) == self.num_variables:
                    # (k_samples, batch, K)
                    # sumlayer = self.splayer_factory.get_variationalsumlayer_activation(scope, self.K,
                    #                                                                    zs=self.LeafLayer.zs, )
                    sumlayer = self.splayer_factory.get_iwsumlayer_activation(scope, self.K)
                else:
                    # sumlayer = self.splayer_factory.get_variationalsumlayer_activation(scope, self.sumK,
                    #                                                                    zs=self.LeafLayer.zs)
                    sumlayer = self.splayer_factory.get_iwsumlayer_activation(scope, self.sumK)

                NUM_SUM_NODES += int(sumlayer.shape[2])

        print("Building SPN took " + str(time.time() - start_time), " seconds")
        print("In total, there are {} leaf nodes, {} sum nodes, and {} product nodes".format(
            NUM_LEAF_NODES, NUM_SUM_NODES, NUM_PROD_NODES))

        avg_likelihood = tf.reduce_logsumexp(sumlayer, axis=0) - tf.log(
            tf.cast(int(sumlayer.shape[0]), tf.float32))  # shape == (batch,)

        return avg_likelihood
