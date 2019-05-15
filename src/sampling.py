import numpy as np

class Map(object):
    def __init__(self, region_graph, sumK=3, leafK=3):
        self.region_graph = region_graph
        self.sumlayers = {} # sumblock.id -> [(s_id, k), ...]
        self.sumK = sumK
        self.partitionlayers = {} # partitionblock.id -> [((s1_id, k1),(s2_id, k2)), ...]
        self.productlayers = {} # frozenset([partitionblock.id]) -> [((s1_id, k1),(s2_id, k2)), ...]
        self.leaflayers = {}  # leafblock.id -> [(s_id, k), ...]
        self.leafK = leafK

        self.leaf_ids = set()

        for leaf_scopeblock in region_graph.get_leaf_scopes():
            self.get_sumlayer_map(leaf_scopeblock)
            self.leaf_ids.add(leaf_scopeblock.id)

        scopes, partitions = region_graph.toposort_traverse_scopes(large2small=False)

        scopes = scopes[1:]

        for partitions_with_same_rank, scopes_with_same_rank in zip(partitions, scopes):

            for partition in partitions_with_same_rank:
                self.get_partitionlayer_map(partition)

            for scope in scopes_with_same_rank:
                self.get_productlayer_map(scope.children)

    def get_sumlayer_map(self, scopeblock):
        """

        :param scopeblock:
        :return: [(scope_id, 0), (scope_id, 1), ...]
        """
        if scopeblock.id in self.sumlayers:
            return self.sumlayers[scopeblock.id]

        else:
            out = [(scopeblock.id, k) for k in range(self.sumK)]
            self.sumlayers[scopeblock.id] = out
            return out

    def get_partitionlayer_map(self, partitionblock):
        """

        :param partitionblock:
        :return: [(A,B), (C,D), ...]
        """
        if partitionblock.id in self.partitionlayers:
            return self.partitionlayers[partitionblock.id]
        else:
            scopeblock1, scopeblock2 = partitionblock.children
            sumlayer1 = self.get_sumlayer_map(scopeblock1)
            sumlayer2 = self.get_sumlayer_map(scopeblock2)

            partitionlayer = ProductOp(sumlayer1, sumlayer2)
            self.partitionlayers[partitionblock.id] = partitionlayer

            return partitionlayer

    def get_productlayer_map(self, list_partitionblock):
        """

        :param list_partitionblock:
        :return: [(A,B), (C,D), ...]
        """


        key = frozenset([partitionblock.id for partitionblock in list_partitionblock])

        if key in self.productlayers:
            return self.productlayers[key]

        else:

            value = []
            for partitionblock in sorted(list_partitionblock, key=lambda x: x.id):
                value += self.get_partitionlayer_map(partitionblock)
            self.productlayers[key] = value

            return value

    def get_leaflayer_map(self, scopeblock):
        if scopeblock.id in self.leaflayers:
            return self.leaflayers[scopeblock.id]
        else:
            out = [(scopeblock.id, k) for k in range(self.sumK)]
            self.leaflayers[scopeblock.id] = out
            return out

    def is_leaf(self, scopeblock_id):
        return scopeblock_id in self.leaf_ids

    def get_root(self):
        return self.region_graph.root

    def get_scope_byid(self, scope_id):
        return self.region_graph.get_scope_byid(scope_id)

class Sampler(object):
    def __init__(self, MAP, get_sumnode_prob):
        """

        :param MAP:
        :param get_sumnode_prob: a function
            (scope_id, k) -> probabilities
        """
        self.get_sumnode_prob = get_sumnode_prob
        self.MAP = MAP
    def descend_sumlayer(self, scopeblock, k):
        """

        :param scopeblock:
        :return: (scope_id1, k1), (scope_id2, k2)
        """
        productnodes = self.MAP.get_productlayer_map(scopeblock.children)
        probabilities = self.get_sumnode_prob(scopeblock.id, k)
        assert len(productnodes) == probabilities.size
        assert np.isclose(np.sum(probabilities), 1)
        i = np.random.choice(np.arange(len(productnodes)), p=probabilities)

        return productnodes[i]

    def sample(self):
        # sample a binary tree

        # recursive strategy
        rootblock = self.MAP.get_root()

        inp = list(self.descend_sumlayer(rootblock, 0))

        descended = 1
        output = []
        while len(inp):

            scope_id, k = inp.pop(0)
            if self.MAP.is_leaf(scope_id):
               output.append((scope_id, k))
            else:

                scopeblock = self.MAP.get_scope_byid(scope_id)
                A, B = self.descend_sumlayer(scopeblock, k)
                inp.append(A)
                inp.append(B)
                descended += 1
                print("descended : {}".format(descended))

        return output

def ProductOp(sumlayer1, sumlayer2):

    """

    :param sumlayer1: [(scope_id, k), (scope_id, k)]
    :param sumlayer2: [(scope_id, k), (scope_id, k)]
    :param partition_id:
    :return:
    """

    out = []
    for i in range(len(sumlayer2)):
        for j in range(len(sumlayer1)):
            out.append((sumlayer1[j], sumlayer2[i]))

    return out
