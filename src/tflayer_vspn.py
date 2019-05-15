import tensorflow as tf
from src.tfdensitynet import MLP2
from src.tflayer_leaf import TFLayer
from src.tfdistributions import concrete_logprob, concrete_sample
import time


class VariationalSumTFLayer(TFLayer):
    """
    # Use SumTFLayer instance as a function
    output = SumTFLayer(5,1234)(prodlayer_input)
    """

    def __init__(self, K, scope_id, temperature, n_hidden=(16)):
        """
        param K: number of nodes in SumLayer (integer)
        param scope_id: (integer)
        """
        super(VariationalSumTFLayer, self).__init__(K, scope_id)
        self.temp = temperature
        self.n_hidden = n_hidden

    def build(self, productlayers, bothlatent):

        """

        :param productlayers:  shape == (k_samples, batch, nprods)
        :param bothlatent: shape == (k_samples, batch, nz)
        :return:
        """
        # K1 = int(productlayers.shape[1])
        # K2 = int(productlayers.shape[2])
        # numlayers = int(productlayers.shape[3])
        #
        # num_nodes = K1 * K2 * numlayers * self.K

        # shape == (k_samples, batch, K^2)

        k_samples = int(productlayers.shape[0])
        batch_size = tf.shape(productlayers)[1]
        nprods = int(productlayers.shape[2])

        print("bothlatent has sape {}".format(bothlatent.shape))

        # directly predict log_p. No probabilistic interpretation
        # gener = MLP2(name=str(self.scope_id), n_hidden=self.n_hidden, n_output=nprods*self.K, transfer_fn=tf.nn.softplus)
        # self.W = tf.reshape(gener(bothlatent), [k_samples, batch_size, nprods, self.K])
        # self.log_p = self.W - tf.expand_dims(tf.reduce_logsumexp(self.W, axis=2) ,axis=2) # shape = (k_samples, batch, K^2, K)

        # concrete distribution. bayesian approach
        gener_alpha = MLP2(name=str(self.scope_id), n_hidden=self.n_hidden, n_output=nprods * self.K,
                     transfer_fn=tf.nn.softplus)
        # print("bothlatent shape", bothlatent.shape)  # shape == (k_samples, batch, nvar)

        random_switch = tf.round(tf.random_uniform(shape=[k_samples, batch_size, 1, self.K]))
        random_switch_complement = 1 - random_switch
        random_log_alpha = tf.random_uniform(shape =[k_samples, batch_size, nprods, self.K])
        log_alpha = tf.check_numerics(tf.nn.sigmoid(tf.reshape(gener_alpha(bothlatent), [k_samples, batch_size, nprods, self.K])), "log_alpha")
        # self.log_alpha = random_log_alpha * random_switch + log_alpha * random_switch_complement
        self.log_alpha = random_log_alpha
        # gener_invtemp = MLP2(name="invtemp"+str(self.scope_id), n_hidden=self.n_hidden, n_output=self.K,
        #              transfer_fn=tf.nn.softplus)
        # self.invtemp = tf.exp(tf.reshape(gener_temp(bothlatent), [k_samples, batch_size, 1, self.K]))


        shape = (k_samples, batch_size, nprods, self.K)
        self.log_p = concrete_sample(shape, self.log_alpha, inv_temp=1./self.temp, log_p=True)  # shape == (k_samples, batch, nprods, K)
        # print("log_alpha shape: {}, temp shape: {}, log_p shape {}".format(self.log_alpha.shape, self.temp.shape, self.log_p.shape))
        # self.lqygivenxz = tf.reduce_sum(concrete_logprob(self.log_p, self.log_alpha, self.temp, axis=2), axis=[2,3])  # shape == (k_samples, batch)
        self.lqygivenxz = tf.ones(dtype=tf.float32, shape=(k_samples, batch_size))
        # print("lqygivenxz", self.lqygivenxz.shape)
        # concrete prior on probability vector
        # using log_alpha = 0
        # using temp = 0.6
        self.lpy = tf.reduce_sum(concrete_logprob(self.log_p, 0., self.temp, axis=2), axis=[2,3])  # shape == (k_samples, batch)
        print("lpy shape: {}, qygivenzx shape: {}".format(self.lpy.shape,self.lqygivenxz.shape))
        self.built = True

    def __call__(self, productlayers, bothlatent):
        """
        multiply weight matrix with child layer
        :param productlayers: list of `Tensor`s of shape == (k_samples, batch, K, K)
        :return: a `Tensor` of shape == (k_samples, batch, K)
        """
        with tf.variable_scope("SumLayer" + str(self.scope_id)):

            # this contains a copy operation !! slow!!
            assert type(productlayers) == list
            assert len(productlayers)>0
            productlayers = tf.concat(productlayers, axis=2)  # shape == (k_samples, batch, K^2)

            if not self.built:
                self.build(productlayers, bothlatent)

            # # top pass of activation
            prods = tf.expand_dims(productlayers, axis=3)  # shape = (k_samples, batch, K^2, 1)
            out = prods + self.log_p  # exploit broadcasting # shape = (k_samples, batch, K^2 K)
            activation = tf.reduce_logsumexp(out, axis=2,
                                             name="activation")  # shape == (k_samples, batch, K)
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


class VariationalSP_TFLayerFactory(object):
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

    def __init__(self, leaflayer_constructor, temperature):
        self.LeafLayer = leaflayer_constructor
        self.temperature = temperature

        # activations
        self._sumlayers_activation = {}  # scopeblock instance id as key
        self._productlayers_activation = {}  # partitionblock instance id as key

        # model
        self.leaflayer = {}
        self.sumlayer = {}

        self.zs_set = set()

        # reconstructions
        self.sumlayer_reconstruction = {}
        self.productlayer_reconstruction = {}

    def get_variationalsumlayer_activation(self, scopeblock, K=None, bothlatent=None, latent_mlp=(16)):
        """
        get sumlayer. evaluate if not yet computed
        param scopeblock: object describing the scope of the sumlayer and its parents and children
        param K: integer, number of nodes in sumlayer
        return: a `Tensor` of shape == (batch, K)
        """
        # Sum Layer already constructed
        if scopeblock.id not in self._sumlayers_activation:
            assert K is not None
            assert bothlatent is not None
            # collect product layers into a sorted list
            child_productlayers = [self.get_productlayer_activation(partitionblock)
                                   for partitionblock in sorted(scopeblock.children, key=lambda x: x.id)]

            # evaluate sum layer
            sl = VariationalSumTFLayer(K, scopeblock.id, temperature=self.temperature, n_hidden=latent_mlp)
            self.sumlayer[scopeblock.id] = sl
            sumlayer = sl(child_productlayers, bothlatent)
            self._sumlayers_activation[scopeblock.id] = sumlayer
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
            leaflayer_activation = leaf(x, average=True)
            # print(leaflayer_activation.shape)
            leaflayer_activation = tf.expand_dims(leaflayer_activation, axis=0)  # shape == (1, batch, K)
            leaflayer_activation = tf.tile(leaflayer_activation, multiples=[leaf.k_samples, 1, 1])  # shape == (k_samples, batch, K)
            self._sumlayers_activation[scope_id] = leaflayer_activation

        if hasattr(leaf, "qz_params"):
            for qz_param in leaf.qz_params:
                self.zs_set.add(qz_param[2])

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

            sumlayer1 = self.get_variationalsumlayer_activation(scopeblock1)
            sumlayer2 = self.get_variationalsumlayer_activation(scopeblock2)

            prodlayer = ProductOp(sumlayer1, sumlayer2, partitionblock.id)
            self._productlayers_activation[partitionblock.id] = prodlayer
        return self._productlayers_activation[partitionblock.id]


class VariationalSPNTFLayer(TFLayer):
    """
    Usage:
    # SPN is a function built recursively from simpler functions
    px = SPNTFLayer(region_graph, BernoulliTFLayer, 3, 3)(x)
    """

    def __init__(self, K, scope_id, region_graph, leaflayer, temperature=tf.Variable(1., trainable=False, dtype=tf.float32), sumK=3, leafK=3, latent_mlp=(16)):
        """
        :param K:
        :param scope_id:
        :param region_graph: architecture / structure of SPN
        :param leaflayer: distribution type of SPN
        :param sumK: number of nodes per sumlayer
        :param leafK: number of nodes per leaflayer
        """

        self.temperature = temperature

        # SPN structure
        self.sumK = sumK
        self.leafK = leafK
        self.region_graph = region_graph
        self.num_variables = len(region_graph.root.vars)

        # SPN distribution type
        self.LeafLayer = leaflayer
        self.latent_mlp = latent_mlp
        self.splayer_factory = VariationalSP_TFLayerFactory(self.LeafLayer, self.temperature)
        super(VariationalSPNTFLayer, self).__init__(K, scope_id)

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

        all_zs = sorted(list(self.splayer_factory.zs_set), key=lambda x:x.name)
        print("Found {} zs tensors".format(len(all_zs)))
        zs_concat = tf.concat(all_zs, axis = 2)  # shape == (k_samples, batch, nz)
        k_samples = int(zs_concat.shape[0])
        x_tiled = tf.tile(tf.expand_dims(x, axis=0), multiples=[k_samples, 1, 1])
        bothlatent = tf.concat([zs_concat, x_tiled], axis = 2)


        start_time = time.time()
        # first list within scopes is a list of leaf scopes.
        # They are already processed. Hence skipped
        scopes = scopes[1:]

        if len(partitions) == 0 or len(scopes) == 0:
            # all variables are under a single leaf layer
            assert len(self.splayer_factory.leaflayer) == 1, "factory leaflayer store has {} entries".format(
                len(self.splayer_factory.leaflayer))
            assert len(self.splayer_factory.sumlayer) == 0, "factory sumlayer store has {} entries".format(
                len(self.splayer_factory.sumlayer))

            arbitrary_scope_id = 10 # choose an integer that does not collide with existing ids

            sl = VariationalSumTFLayer(1, arbitrary_scope_id, self.temperature, self.latent_mlp)
            leaflayer_activation = next(iter(self.splayer_factory._sumlayers_activation.values()))

            sumlayer_activation = sl([leaflayer_activation], bothlatent)
            self.splayer_factory.sumlayer[arbitrary_scope_id] = sl
            self.splayer_factory._sumlayers_activation[arbitrary_scope_id] = sumlayer_activation

        else:
            if self.leafK == 1:
                # equivalent to an IWAE
                # root likelihood is a product of all leaves
                # I am going to disregard the scopeblock graph
                leaflayer_activations = self.splayer_factory._sumlayers_activation.values() # list of tensors of shape (k_samples, batch)
                sumlayer_activation = tf.add_n(leaflayer_activations) # multiplication in real space is addition in log space
            else:
                for partitions_with_same_rank, scopes_with_same_rank in zip(partitions, scopes):
                    print(str(len(partitions_with_same_rank)) + " partitions, " + str(len(scopes_with_same_rank)) + " scopes")

                    for partition in partitions_with_same_rank:
                        productlayer = self.splayer_factory.get_productlayer_activation(partition)  # (k_samples, batch, K)
                        NUM_PROD_NODES += int(productlayer.shape[2])

                    for scope in scopes_with_same_rank:
                        # testing if root scope
                        if len(scope.vars) == self.num_variables:
                            # we have reached the root
                            # (k_samples, batch, K)
                            sumlayer_activation = self.splayer_factory.get_variationalsumlayer_activation(scope, self.K, bothlatent, self.latent_mlp)
                        else:
                            sumlayer_activation = self.splayer_factory.get_variationalsumlayer_activation(scope, self.sumK, bothlatent, self.latent_mlp)

                        NUM_SUM_NODES += int(sumlayer_activation.shape[2])

                print("Building SPN took " + str(time.time() - start_time), " seconds")
                print("In total, there are {} leaf nodes, {} sum nodes, and {} product nodes".format(
                    NUM_LEAF_NODES, NUM_SUM_NODES, NUM_PROD_NODES))


        # log p(x,y|z) = log p(x|y,z) * p(y)
        lpxgivenyz = tf.squeeze(sumlayer_activation) # shape == (k_samples, batch)
        lpy = tf.add_n([sl.lpy for scope_id, sl in self.splayer_factory.sumlayer.items()])

        # log p(
        log_numerator = tf.add(lpxgivenyz, lpy)  # shape == (k_samples, batch,)

        # log q(y|x,z)
        log_denominator = tf.add_n([sl.lqygivenxz for scope_id, sl in self.splayer_factory.sumlayer.items()])
        # shape == (k_samples, batch,)

        # importance weighting of samples
        samples_liklihood = tf.subtract(log_numerator, log_denominator)  # shape == (k_samples, batch,)

        avg_likelihood = tf.reduce_logsumexp(samples_liklihood, axis=0) - tf.log(
            tf.cast(int(sumlayer_activation.shape[0]), tf.float32))  # shape == (batch,)

        return avg_likelihood
