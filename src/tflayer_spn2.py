"""
TFLayer implementation of Poon Domingos SPN

sum weights have shape [1, K1, K2, numlayers, K]
"""

import tensorflow as tf
from src.tflayer_leaf import BernoulliTFLayer, TFLayer
import time

class SumTFLayer(TFLayer):
    """
    # Use SumTFLayer instance as a function
    output = SumTFLayer(5,1234)(prodlayer_input)
    """
    def __init__(self, K, scope_id, debug=False):
        """
        param K: number of nodes in SumLayer (integer)
        param scope_id: (integer)
        """
        super(SumTFLayer, self).__init__(K, scope_id)
        self.debug = debug

    def build(self, productlayers):
        """
        associate a weight matrix with sumlayer
        """
        K1 = int(productlayers.shape[1])
        K2 = int(productlayers.shape[2])
        numlayers = int(productlayers.shape[3])

        if self.debug:
            self.W = tf.constant(0, name="weights",
                                 shape=[1, K1, K2, numlayers, self.K],
                                 dtype=tf.float32,
                                 )  # shape = (1, K1,K2,numlayers, K)
        else:
            self.W = tf.get_variable(name="weights",
                                     shape=[1, K1, K2, numlayers, self.K],
                                     dtype=tf.float32,
                                     initializer=tf.initializers.random_normal(),
                                     regularizer=tf.contrib.layers.l1_regularizer(0.1),
                                     )  # shape = (1, K1,K2,numlayers, K)

        self.log_p = self.W - tf.reduce_logsumexp(self.W, axis=[1,2,3])  # shape = (1, K1,K2,numlayers, K)
        # log_p = tf.expand_dims(log_p, axis=0)  # shape = (1, K^2, K)
        self.built = True

    def __call__(self, productlayers):
        """
        multiply weight matrix with child layer
        :param productlayers: list of `Tensor`s of shape == (batch, K, K)
        :return: a `Tensor` of shape == (batch, K)
        """
        with tf.variable_scope("SumLayer" + str(self.scope_id)):

            # this contains a copy operation !! slow!!
            productlayers = tf.stack(productlayers, axis=3) # shape == (batch, K, K, numlayers)

            if not self.built:
                self.build(productlayers)

            # # top pass of activation
            prods = tf.expand_dims(productlayers, axis=4)  # shape = (1, K1,K2,numlayers, 1)
            if len(prods.shape) > 5:
                # we are passing in probability maps
                log_p = tf.expand_dims(self.log_p, axis=5)  # shape = (1, K1,K2,numlayers, K, 1)
            elif len(prods.shape) == 5:
                log_p = self.log_p
            else:
                raise ValueError
            out = prods + log_p  # exploit broadcasting # shape = (1, K1,K2,numlayers, K)  or (1, K1,K2,numlayers, K, vars)
            activation = tf.reduce_logsumexp(out, axis=[1,2,3], name="activation")  # shape == (batch, K)  or (batch, K, vars)

            return activation

    def reconstruction(self, productlayers):
        indices = tf.argmax(self.W, axis=[])

    def get_node_entropy(self):
        entropy = tf.reduce_sum(-tf.exp(self.log_p) * self.log_p)
        return entropy


    def get_layer_entropy(self):
        log_p_bar = self.log_p - tf.expand_dims(tf.reduce_logsumexp(self.log_p, axis=[4]),axis=4)
        entropy = tf.reduce_sum(-tf.exp(log_p_bar) * log_p_bar)
        return entropy



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
        sl1 = tf.expand_dims(sumlayer1, axis=1)  # shape = (batchsize, 1, K)
        sl2 = tf.expand_dims(sumlayer2, axis=2)  # shape = (batchsize, K, 1)

        # catesian product in log domain
        # exploit broadcasting
        batch_outer = tf.add(sl1, sl2, name = "activation")  # shape = (batchsize, K, K)
        return batch_outer


# This is not an important function
# I was experimenting if a probability map can be passed upwards
# through an SPN
def ProductProbMap(probmap1, probmap2, partition_id, var1, var2):

    with tf.variable_scope("ProdLayer" + str(partition_id)):
        # top pass of activation
        sl1 = tf.expand_dims(probmap1, axis=1)  # shape = (batchsize, 1, K1, var1)
        sl2 = tf.expand_dims(probmap2, axis=2)  # shape = (batchsize, K2, 1, var2)

        K2 = int(sl2.shape[1])
        K1 = int(sl1.shape[2])

        sl1 = tf.keras.backend.repeat_elements(sl1, K2, axis=1) # shape = (batchsize, K2, K1, var1)
        sl2 = tf.keras.backend.repeat_elements(sl2, K1, axis=2) # shape = (batchsize, K2, K1, var2)

        sl = tf.concat([sl1, sl2], axis=3) # shape = (batchsize, K2, K1, var3)

        vars = var1 + var2
        indices = sorted(range(len(vars)), key=lambda k: vars[k])
        sl = tf.gather(sl,indices, axis = 3)

        return sl


class SP_TFLayerFactory(object):
    """
    This class encapsulates the creation and storage of SumTFLayer and ProductTFLayer.
    Every SumTFLayer represents K distributions over some set of random variables
    Every ProductTFLayer represents K^2 distributions over some set of random variables

    Block Graph handles the indexing and relationships between ScopeBlocks and PartitionBlocks

    ScopeBlocks and PartitionBlocks are thus used as construction arguments

    SumTFLayers are indexed by their associated ScopeBlock instance id which is an integer

    ProductTFLayers are indexed by their associated PartitionBlock instance id which is an integer

    I had tried using strings as unqiue names to index SumTFLayers and ProductTFLayers
    But there are a lot of them and having interpretable string names may create too much
    overhead in tensorflow naming system
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

        self.sumnode_entropy = 0
        self.layer_entropy = 0

    def get_sumlayer_activation(self, scopeblock, K=None):
        """
        get sumlayer. evaluate if not yet computed
        param scopeblock: object describing the scope of the sumlayer and its parents and children
        param K: integer, number of nodes in sumlayer
        return: a `Tensor` of shape == (batch, K)
        """
        # Sum Layer already constructed
        if scopeblock.id not in self._sumlayers_activation:
            assert K is not None
            # collect product layers into a sorted list
            child_productlayers = [self.get_productlayer_activation(partitionblock)
                                   for partitionblock in sorted(scopeblock.children, key=lambda x: x.id)]

            # evaluate sum layer
            sl = SumTFLayer(K, scopeblock.id)
            self.sumlayer[scopeblock] = sl
            sumlayer = sl(child_productlayers)
            self._sumlayers_activation[scopeblock.id] = sumlayer

            self.sumnode_entropy+= sl.get_node_entropy()
            self.layer_entropy += sl.get_layer_entropy()

        return self._sumlayers_activation[scopeblock.id]


    def get_leaflayer_activation(self, x, K, scope_id):
        """
        get leaflayer. evaluate if not yet computed
        param x: input data, `Tensor` of shape == (batch, num_var)
        param K: number of nodes in leaflayer, integer
        param scope_id: integer, Every leaflayer has a unique scope
        return: a `Tensor` of shape == (batch, K)
        """

        if scope_id not in self._sumlayers_activation:
            leaf = self.LeafLayer(K, scope_id)
            self.leaflayer[scope_id] = leaf
            leaflayer = leaf(x)
            self._sumlayers_activation[scope_id] = leaflayer

        return self._sumlayers_activation[scope_id]

    def get_productlayer_activation(self, partitionblock):
        """
        get productlayer. evaluate if not yet computed
        param partitionblock: object describing the scope of the productlayer, its parents and children
        return: a `Tensor` of shape == (batch, K^2)
        """

        if partitionblock.id not in self._productlayers_activation:
            assert len(partitionblock.children) == 2
            scopeblock1, scopeblock2 = partitionblock.children

            sumlayer1 = self.get_sumlayer_activation(scopeblock1)
            sumlayer2 = self.get_sumlayer_activation(scopeblock2)

            prodlayer = ProductOp(sumlayer1, sumlayer2, partitionblock.id)
            self._productlayers_activation[partitionblock.id] = prodlayer

        return self._productlayers_activation[partitionblock.id]

    def get_leaflayer_reconstruction(self, scope_id):
        """
        get leaflayer. evaluate if not yet computed
        param x: input data, `Tensor` of shape == (batch, num_var)
        param K: number of nodes in leaflayer, integer
        param scope_id: integer, Every leaflayer has a unique scope
        return: a `Tensor` of shape == (batch, K)
        """

        if scope_id in self.sumlayer_reconstruction:
            return self.sumlayer_reconstruction[scope_id]
        else:
            assert scope_id in self.leaflayer
            x_reconstr_means = self.leaflayer[scope_id].x_reconstr_means
            self.sumlayer_reconstruction[scope_id] = x_reconstr_means
            return x_reconstr_means

    def get_sumlayer_reconstruction(self, scopeblock):
        # Sum Layer already constructed
        if scopeblock.id in self.sumlayer_reconstruction:
            return self.sumlayer_reconstruction[scopeblock.id]
        # Sum Layer not yet constructed
        else:
            assert scopeblock.id in self.sumlayer
            child_productlayers = [self.get_productlayer_reconstruction(partitionblock)
                                   for partitionblock in sorted(scopeblock.children, key=lambda x: x.id)]
            sl = self.sumlayer[scopeblock.id]

            x_reconstr_means = sl(child_productlayers)
            self.sumlayer_reconstruction[scopeblock.id] = x_reconstr_means
            return x_reconstr_means


    def get_productlayer_reconstruction(self, partitionblock):
        """
        get productlayer. evaluate if not yet computed
        param partitionblock: object describing the scope of the productlayer, its parents and children
        return: a `Tensor` of shape == (batch, K^2)
        """

        if partitionblock.id in self.productlayer_reconstruction:
            return self.productlayer_reconstruction[partitionblock.id]
        else:
            assert len(partitionblock.children) == 2
            scopeblock1, scopeblock2 = partitionblock.children

            sumlayer1 = self.get_sumlayer_reconstruction(scopeblock1)
            sumlayer2 = self.get_sumlayer_reconstruction(scopeblock2)

            prodlayer = ProductProbMap(sumlayer1, sumlayer2, partitionblock.id, sorted(scopeblock1.vars), sorted(scopeblock2.vars))
            self.productlayer_reconstruction[partitionblock.id] = prodlayer
            return prodlayer



class SPNTFLayer(TFLayer):
    """
    Usage:
    # SPN is a function built recursively from simpler functions
    px = SPNTFLayer(region_graph, BernoulliTFLayer, 3, 3)(x)
    """
    def __init__(self, K, scope_id, region_graph, leaflayer=BernoulliTFLayer, sumK=3, leafK=3):
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
        self.splayer_factory = SP_TFLayerFactory(self.LeafLayer)
        super(SPNTFLayer, self).__init__(K, scope_id)

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
            leaflayer = self.splayer_factory.get_leaflayer_activation(x=x_subset, K=self.leafK, scope_id=leaf_scope.id)
            NUM_LEAF_NODES += int(leaflayer.shape[1])

        print("start traversal")
        scopes, partitions = self.region_graph.toposort_traverse_scopes(large2small=False)
        print("end traversal")

        start_time = time.time()
        # first list within scopes is a list of leaf scopes.
        # They are already processed. Hence skipped
        scopes = scopes[1:]

        if len(partitions) == 0 or len(scopes) == 0:
            # all variables are under the same leaf layer
            # equivalent to a mixture of IWAEs
            assert len(self.splayer_factory.leaflayer) == 1, "factory leaflayer store has {} entries".format(
                len(self.splayer_factory.leaflayer))
            assert len(self.splayer_factory.sumlayer) == 0, "factory sumlayer store has {} entries".format(
                len(self.splayer_factory.sumlayer))

            arbitrary_scope_id = 10  # choose an integer that does not collide with existing ids

            sl = SumTFLayer(1, arbitrary_scope_id)
            leaflayer_activation = next(iter(self.splayer_factory._sumlayers_activation.values()))

            sumlayer_activation = sl([leaflayer_activation])
            self.splayer_factory.sumlayer[arbitrary_scope_id] = sl
            self.splayer_factory._sumlayers_activation[arbitrary_scope_id] = sumlayer_activation

        else:
            if self.leafK == 1:
                # equivalent to a product of IWAEs
                # root likelihood is a product of all leaves
                # I am going to disregard the scopeblock graph
                leaflayer_activations = self.splayer_factory._sumlayers_activation.values() # list of tensors of shape (batch)
                sumlayer_activation = tf.add_n(leaflayer_activations) # multiplication in real space is addition in log space
            else:
                for partitions_with_same_rank, scopes_with_same_rank in zip(partitions, scopes):
                    print(str(len(partitions_with_same_rank)) + " partitions, " + str(len(scopes_with_same_rank)) + " scopes")
                    for partition in partitions_with_same_rank:
                        productlayer_activation = self.splayer_factory.get_productlayer_activation(partition)
                        NUM_PROD_NODES += int(productlayer_activation.shape[1])

                    for scope in scopes_with_same_rank:
                        # testing if root scope
                        if len(scope.vars) == self.num_variables:
                            sumlayer_activation = self.splayer_factory.get_sumlayer_activation(scope, self.K)
                        else:
                            sumlayer_activation = self.splayer_factory.get_sumlayer_activation(scope, self.sumK)
                        NUM_SUM_NODES += int(sumlayer_activation.shape[1])

                print("Building SPN took " + str(time.time() - start_time), " seconds")
                print("In total, there are {} leaf nodes, {} sum nodes, and {} product nodes".format(
                    NUM_LEAF_NODES, NUM_SUM_NODES, NUM_PROD_NODES))

        tf.summary.scalar(name="node_entropy", tensor=self.splayer_factory.sumnode_entropy)
        tf.summary.scalar(name="layer_entropy", tensor=self.splayer_factory.layer_entropy)

        return sumlayer_activation