from collections import deque
from abc import ABC
from itertools import groupby
import numpy as np


class Block(ABC):
    def __init__(self, vars1, vars2=None):
        # instance id to be set by factory
        self._id = None

        # keys for hashing and naming
        if vars2:  # PartitionBlock
            self._vars = frozenset([vars1, vars2])
        else:  # ScopeBlock
            self._vars = frozenset(vars1)

        # links for graph traversal
        self._parents = set()
        self._children = set()

    @property
    def id(self):
        if self._id is None:
            raise NotImplementedError
        else:
            return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    @property
    def vars(self):
        return self._vars


class ScopeBlock(Block):
    """
    The set of random variables over which a probability distribution is defined

    Every ScopeBlock may have 0 to many PartitionBlock children
    Every ScopeBlock may have 0 to many PartitionBlock parents
    """

    def __init__(self, random_variables):
        """
        ScopeBlock is nothing more than a wrapper for a set of random variables.
        Using frozenset because it is immutable and thus can be hashed
        Do not construct ScopeBlocks yourself. Use the factory design pattern
        """

        super(ScopeBlock, self).__init__(random_variables)

    def disjoint(self, other_scope):
        return bool(self.vars & other_scope.vars)

    def is_atomic(self):
        return len(self.vars) == 1

    def __repr__(self):
        var_str = ",".join([str(var) for var in self.vars])
        part_str = ",".join([str(p.id) for p in self.children])
        # s means scopeblock
        return "(s:{0}, vars:{{{1}}}, p:[{2}])\n".format(self.id, var_str, part_str)


class RegionBlock(ScopeBlock):
    """
    A Region is a Scope(Subset) made by r.v.s that are spatially
    adjacent
    """

    @classmethod
    def id_from_coords(cls, i, j, n_cols):
        return i * n_cols + j

    @classmethod
    def from_region_to_var_ids(cls, a_1, a_2, b_1, b_2, tot_n_cols):
        """
        Translating a Region coordinate represntation
        to an id representation (set of ids)
        """

        n_cols = b_2 - b_1
        n_rows = a_2 - a_1
        starting_id = RegionBlock.id_from_coords(a_1, b_1, tot_n_cols)
        variables = set()
        for i in range(n_rows):
            for j in range(n_cols):
                variables.add(starting_id + j)
            starting_id += tot_n_cols
        return variables

    def __init__(self, x_1, x_2, y_1, y_2, image_n_rows, image_n_cols):
        """
        A region in an image (n_rows X n_cols) can be represented by 4 coordinates:
        (x_1, y_1) (x_2, y_2)
        """

        # type checking
        assert all([isinstance(x, int) for x in [x_1, x_2, y_1, y_2]])

        self.x_1, self.y_1 = x_1, y_1
        self.x_2, self.y_2 = x_2, y_2

        self.width = y_2 - y_1
        self.height = x_2 - x_1

        assert self.width > 0
        assert self.height > 0

        # why is this needed ???
        self.image_n_rows = image_n_rows
        self.image_n_cols = image_n_cols

        variables = RegionBlock.from_region_to_var_ids(x_1, x_2, y_1, y_2, image_n_cols)
        super().__init__(variables)

    def is_fine_region(self, base_res):
        """If image is smaller than coarsegrain threshold,
        it is considered a fine region. Otherwise it is
        a coarse region"""
        return self.width <= base_res and self.height <= base_res

    def __repr__(self):
        var_str = ",".join([str(var) for var in self.vars])
        part_str = ",".join([str(p.id) for p in self.children])
        # "r" means region
        return "(r:{0}, [{1}, {2}, {3}, {4}], vars:{{{5}}}, p:[{6}])\n".format(self.id,
                                                                               self.x_1,
                                                                               self.y_1,
                                                                               self.x_2,
                                                                               self.y_2,
                                                                               var_str,
                                                                               part_str)


class PartitionBlock(Block):
    """
    A partition over a scope, represented as a sequence of disjoint Scope(Subset)s
    Plus a global Scope(Subset) as the union of all Scope(Subset)s

    Every Partition has one parent scope subset
    """

    def __init__(self, children_scopes, parent_scope):
        """
        PartitionBlock is nothing more than a bag of parent and child links to ScopeBlock
        Partition of parent scope is a pair of disjoint children scopes objects

        Do not construct ScopePartition objects yourself. Use the Factory design pattern
        :param children_scopes: set of disjoint scopes objects
        """

        # sanity check inputs
        assert (len(children_scopes)) == 2
        child_scope1, child_scope2 = children_scopes
        super(PartitionBlock, self).__init__(child_scope1.vars, child_scope2.vars)

        # links to children scope objects
        self.children.update(children_scopes)
        # partition block tells the child scope block: "I am your parent"
        for s in self.children:
            s.parents.add(self)

        # links to parent scope object
        self.parents.add(parent_scope)  # type change from set to singleton
        # partition node tells the parent scope: "I am your child"
        next(iter(self.parents)).children.add(self)

        self.sanity_check()

    def sanity_check(self):
        assert len(self.parents) == 1
        assert len(self.children) > 1

        _parent_scope_rvs = set()

        for s in self.children:
            # check disjointness
            if s.vars & _parent_scope_rvs:
                raise ValueError('Creating a partition over non-disjoint scopes',
                                 s, _parent_scope_rvs)
            _parent_scope_rvs = _parent_scope_rvs.union(s.vars)

        assert next(iter(self.parents)).vars == frozenset(_parent_scope_rvs)

    def __repr__(self):
        scopes_str = ",".join(str(s.id) for s in self.children)
        # "p" means partition
        return "(id:{0} p:<{1}> s:<{{{2}}}>)\n".format(self.id, next(iter(self.parents)).id, scopes_str)


class ScopeBlockFactory(object):
    """
    This class encapsulates the creations mechanism of ScopeBlock objects.
    It ensures that every ScopeBlock is has a unique set of random variables

    There are two ways to index this unique collection of ScopeBlock instances:
        1) instance ids
        2) frozen set of random variables ids

    Instance ids is a more compact way to name each ScopeBlock but it is less interpretable
    """

    def __init__(self):
        self._instance_id = 0
        self._vars2instance = {}  # dict mapping set of random variables to scope objects
        self._id2vars = {}  # dict mapping instance id to set of random variables

    def get_instance(self, random_variables):
        """Ensures that every ScopeBlock object instantiated has a unique set of random_variables"""
        key = frozenset(random_variables)

        if key in self._vars2instance:
            return self._vars2instance[key]
        else:
            obj = ScopeBlock(random_variables)
            obj.id = self._instance_id
            self._vars2instance[key] = obj
            self._id2vars[self._instance_id] = key
            self._instance_id += 1
            return obj

    def get_region_instance(self, x_1, x_2, y_1, y_2, image_n_rows, image_n_cols):
        key = frozenset(RegionBlock.from_region_to_var_ids(x_1, x_2, y_1, y_2, image_n_cols))

        if key in self._vars2instance:
            return self._vars2instance[key]
        else:
            # object construction
            obj = RegionBlock(x_1, x_2, y_1, y_2, image_n_rows, image_n_cols)
            obj.id = self._instance_id

            # indexing object
            self._vars2instance[key] = obj
            self._id2vars[self._instance_id] = key

            # update factory state
            self._instance_id += 1
            return obj

    def get_instance_byid(self, ID):
        key = self._id2vars[ID]
        return self._vars2instance[key]


class PartitionBlockFactory(object):
    """
    This class encapsulates the creations mechanism of PartitionBlock objects.
    It ensures that every PartitionBlock is a unique partition of a parent scope

    There are three ways to index this collection of unique PartitionBlock instances:
        1) instance ids
        2) frozen set of frozenset of child random variable ids
        3) sorted tuple of child scope instance ids
    """

    def __init__(self):
        self._instance_id = 0
        self._vars2instance = {}
        self._id2vars = {}
        self._tuple2id = {}

    def get_instance(self, children_scopes, parent_scope):
        # design choice:
        # at the moment, I enforce that partition block only has 2 children
        # theoretically a region graph can partition into as many children scopes as it needs
        # remove assertion to remove this design choice
        assert (len(children_scopes)) == 2

        key = frozenset([cs.vars for cs in children_scopes])

        if key in self._vars2instance:
            return self._vars2instance[key]
        else:
            # constructing object
            obj = PartitionBlock(children_scopes, parent_scope)
            obj.id = self._instance_id

            # indexing object
            self._vars2instance[key] = obj
            self._id2vars[self._instance_id] = key
            self._tuple2id[tuple(sorted([cs.id for cs in children_scopes]))] = self._instance_id

            # update factory state
            self._instance_id += 1
            return obj

    def get_instance_byid(self, ID):
        key = self._id2vars[ID]
        return self._vars2instance[key]


class BlockGraph(object):
    """
    Block graph is a collection of ScopeBlock and PartitionBlock objects.

    Links between ScopeBlock and PartitionBlock objects form a bipartite graph

    Does not build the bipartite graph. Construction left to subclass
    Only provides utility functions for sanity testing and traversing graph

    This equals to the RegionGraph in Dennis2012
    """

    def __init__(self, root_scope):
        """
        Initialised ScopeGraph with root scope
        """

        self.root = root_scope

        self.scopes = set()
        self.partitions = set()

    def sanity_check_scope(self, scope):
        # ensure the children partition block is linked to current scope block
        assert all([next(iter(partition.parents)) == scope for partition in scope.children])

        # ensure that scope block has legitimate children
        for partition in scope.children:
            vars1, vars2 = partition.vars
            assert vars1 | vars2 == scope.vars

    def sanity_check_partition(self, partition):
        # ensure parent scope block is linked to current partition block
        # every partition block has only 1 parent scope block
        assert partition in next(iter(partition.parents)).children

        # ensure children scope block is linked to current partition block
        assert all([partition in child.parents for child in partition.children])

    def add_scope(self, scope):
        self.scopes.add(scope)
        return scope

    def add_partition(self, partition):
        self.partitions.add(partition)

        return partition

    def get_leaf_scopes(self):
        return [s for s in self.scopes if len(s.children) == 0]

    def traverse_scopes(self, root_scope=None, yield_partitions=False, order='bfs'):
        """
        A generator of scopes and partition objects (optional) in breadth-first search order
        or depth-first search order
        :param root_scope:
        :param yield_partitions:
        :param order: 'bfs' or 'dfs'
        :return:
        """

        # starting point
        if not root_scope:
            root_scope = self.root

        # work queue
        scopes_to_process = deque()
        scopes_to_process.append(root_scope)

        # keeping track
        visited_scopes = set()
        visited_scopes.add(root_scope)

        # python style strategy pattern?
        def enqueue_scope(scope):
            scopes_to_process.append(scope)

        def stack_scope(scope):
            scopes_to_process.appendleft(scope)

        if order == 'bfs':
            add_scope = enqueue_scope
        elif order == 'dfs':
            add_scope = stack_scope
        else:
            raise ValueError('Invalid traversing order', order)

        # main loop
        while scopes_to_process:

            current_scope = scopes_to_process.popleft()
            assert isinstance(current_scope, ScopeBlock)
            yield current_scope  # what a beast

            for current_partition in current_scope.children:
                if yield_partitions:
                    yield current_partition

                for part_scope in current_partition.children:
                    if part_scope not in visited_scopes:
                        add_scope(part_scope)
                        visited_scopes.add(part_scope)

    def toposort_traverse_scopes(self, large2small=False):
        sorted_scopes = sorted(list(self.scopes), key=lambda s: len(s.vars), reverse=large2small)
        sorted_partitions = sorted(list(self.partitions), key=lambda p: len(next(iter(p.parents)).vars),
                                   reverse=large2small)

        out_scopes = []
        for k, v in groupby(sorted_scopes, key=lambda s: len(s.vars)):
            out_scopes.append(list(v))

        out_partitions = []
        for k, v in groupby(sorted_partitions, key=lambda p: len(next(iter(p.parents)).vars)):
            out_partitions.append(list(v))

        return out_scopes, out_partitions

    # Some counting functions
    def n_nodes(self):
        return len(self.partitions) + len(self.scopes)

    def n_scopes(self):
        return len(self.scopes)

    def n_partitions(self):
        return len(self.partitions)

    # Some existence functions
    def is_scope_present(self, scope):
        return scope in self.scopes

    def is_partition_present(self, partition):
        return partition in self.partitions

    def __eq__(self, scope_graph):

        # Well done! I would never have computed equality of graphs :O
        equal = True
        for node_1, node_2 in zip(self.traverse_scopes(yield_partitions=True),
                                  scope_graph.traverse_scopes(yield_partitions=True)):
            equal = node_1 == node_2
            if not equal:
                break
        return equal

    def __repr__(self):
        trav_repr = " ".join([str(node) for node in self.traverse_scopes(order='bfs',
                                                                         yield_partitions=True)])
        return trav_repr


class PoonRegionGraph(BlockGraph):
    def __init__(self, n_rows, n_cols, coarse):
        self.partition_factory = PartitionBlockFactory()
        self.scope_factory = ScopeBlockFactory()
        root_region = self.scope_factory.get_region_instance(0, n_rows, 0, n_cols, n_rows, n_cols)

        super(PoonRegionGraph, self).__init__(root_region)
        self.create_region_graph(coarse)

    def vertically_partition(self, region, skip=1):
        assert skip > 0

        results = []  # list of tuples of (left_region, right_region, parent partition)

        for i in range(region.y_1 + skip, region.y_2, skip):
            # create two new regions
            left_region = self.scope_factory.get_region_instance(region.x_1,
                                                                 region.x_2,
                                                                 region.y_1,
                                                                 i,
                                                                 region.image_n_rows,
                                                                 region.image_n_cols)
            right_region = self.scope_factory.get_region_instance(region.x_1,
                                                                  region.x_2,
                                                                  i,
                                                                  region.y_2,
                                                                  region.image_n_rows,
                                                                  region.image_n_cols)

            partition = self.partition_factory.get_instance(children_scopes={left_region, right_region},
                                                            parent_scope=region)

            # sanity checking
            assert next(iter(partition.parents)).id == region.id

            results.append((left_region, right_region, partition))

        return results

    def horizontally_partition(self, region, skip=1):
        assert skip > 0

        results = []  # list of tuples of (top_region, bottom_region, parent partition)

        for i in range(region.x_1 + skip, region.x_2, skip):
            # create two new regions
            top_region = self.scope_factory.get_region_instance(region.x_1,
                                                                i,
                                                                region.y_1,
                                                                region.y_2,
                                                                region.image_n_rows,
                                                                region.image_n_cols)

            bottom_region = self.scope_factory.get_region_instance(i,
                                                                   region.x_2,
                                                                   region.y_1,
                                                                   region.y_2,
                                                                   region.image_n_rows,
                                                                   region.image_n_cols)

            partition = self.partition_factory.get_instance(children_scopes={top_region, bottom_region},
                                                            parent_scope=region)

            # sanity checking
            assert next(iter(partition.parents)).id == region.id

            results.append((top_region, bottom_region, partition))

        return results

    def choose_coarsegrainlevel(self, region, coarsegrains):
        """

        :param region:
        :param coarsegrains: ordered from small to large
        :return:
        """
        for cgl in reversed(coarsegrains):
            if not region.is_fine_region(cgl):
                return cgl

        return coarsegrains[0]

    def create_region_graph(self, coarsegrain_levels):
        """
        :param coarsegrain_levels: sorted list of coarsegrain level. Example: [1,4,16]
        :return:
        """

        region = self.root
        self.scopes.add(region)  # region_graph also keeps track of which scope nodes have been constructed

        # initialise queue
        regions_to_process = deque()
        regions_to_process.append(region)

        while regions_to_process:
            # get a region to process
            current_region = regions_to_process.popleft()

            coarsegrain_level = self.choose_coarsegrainlevel(current_region, coarsegrain_levels)

            # get all possible decompositions horizontally and vertically
            vertical_region_splits = self.vertically_partition(current_region, skip=coarsegrain_level)
            horizontal_region_splits = self.horizontally_partition(current_region, skip=coarsegrain_level)
            regions_to_consider = vertical_region_splits + horizontal_region_splits

            for region_1, region_2, partition in regions_to_consider:
                # sanity check: constructed partition should always be new
                assert not partition in self.partitions
                self.add_partition(partition)

                self.sanity_check_scope(region_1)
                self.sanity_check_scope(region_2)

                # if freshly constructed, add to queue
                if not region_1 in self.scopes:
                    self.scopes.add(region_1)
                    regions_to_process.append(region_1)
                if not region_2 in self.scopes:
                    self.scopes.add(region_2)
                    regions_to_process.append(region_2)


    def get_scope_byid(self, ID):
        return self.scope_factory.get_instance_byid(ID)

    def get_partition_byid(self, ID):
        return self.partition_factory.get_instance_byid(ID)

class ConvRegionGraph(BlockGraph):
    def __init__(self, n_rows, n_cols, stride, n_rows_fg, n_cols_fg, univariate_bg=True):
        self.partition_factory = PartitionBlockFactory()
        self.scope_factory = ScopeBlockFactory()
        self.n_rows_fg, self.n_cols_fg = n_rows_fg, n_cols_fg
        self.univariate_bg = univariate_bg
        root_region = self.scope_factory.get_region_instance(0, n_rows, 0, n_cols, n_rows, n_cols)

        super(ConvRegionGraph, self).__init__(root_region)
        self.create_region_graph(stride, n_rows_fg, n_cols_fg)

    def vertically_partition(self, region, stride, fg_n_cols):
        assert stride > 0

        results = []  # list of tuples of (left_region, right_region, parent partition)
        # TODO: check if cols and rows are properly matched
        for i in range(region.y_1 + stride, region.y_2, stride):
            left_width = i
            right_width = region.y_2 - i
            if left_width < fg_n_cols and right_width < fg_n_cols:
                continue
            # create two new regions
            left_region = self.scope_factory.get_region_instance(region.x_1,
                                                                 region.x_2,
                                                                 region.y_1,
                                                                 i,
                                                                 region.image_n_rows,
                                                                 region.image_n_cols)
            right_region = self.scope_factory.get_region_instance(region.x_1,
                                                                  region.x_2,
                                                                  i,
                                                                  region.y_2,
                                                                  region.image_n_rows,
                                                                  region.image_n_cols)

            partition = self.partition_factory.get_instance(children_scopes={left_region, right_region},
                                                            parent_scope=region)

            # sanity checking
            assert next(iter(partition.parents)).id == region.id

            results.append((left_region, right_region, partition))

        return results

    def horizontally_partition(self, region, stride, fg_n_rows):
        assert stride > 0

        results = []  # list of tuples of (top_region, bottom_region, parent partition)

        for i in range(region.x_1 + stride, region.x_2, stride):
            top_height = i
            bottom_height = region.x_2 - i
            if top_height < fg_n_rows and bottom_height < fg_n_rows:
                continue
            # create two new regions
            top_region = self.scope_factory.get_region_instance(region.x_1,
                                                                i,
                                                                region.y_1,
                                                                region.y_2,
                                                                region.image_n_rows,
                                                                region.image_n_cols)

            bottom_region = self.scope_factory.get_region_instance(i,
                                                                   region.x_2,
                                                                   region.y_1,
                                                                   region.y_2,
                                                                   region.image_n_rows,
                                                                   region.image_n_cols)

            partition = self.partition_factory.get_instance(children_scopes={top_region, bottom_region},
                                                            parent_scope=region)

            # sanity checking
            assert next(iter(partition.parents)).id == region.id

            results.append((top_region, bottom_region, partition))

        return results

    def random_vertically_partition(self, region, num_partitions):

        results = []  # list of tuples of (left_region, right_region, parent partition)
        assert num_partitions > 0
        total_possible_partitions = region.y_2 - region.y_1 - 1
        if total_possible_partitions > 0:
            num_partitions = min(num_partitions, total_possible_partitions)
            for i in np.random.choice(total_possible_partitions, num_partitions):
                i = int(i + 1)
                # create two new regions
                left_region = self.scope_factory.get_region_instance(region.x_1,
                                                                     region.x_2,
                                                                     region.y_1,
                                                                     region.y_1 + i,
                                                                     region.image_n_rows,
                                                                     region.image_n_cols)
                right_region = self.scope_factory.get_region_instance(region.x_1,
                                                                      region.x_2,
                                                                      region.y_1 + i,
                                                                      region.y_2,
                                                                      region.image_n_rows,
                                                                      region.image_n_cols)

                partition = self.partition_factory.get_instance(children_scopes={left_region, right_region},
                                                                parent_scope=region)

                # sanity checking
                assert next(iter(partition.parents)).id == region.id

                results.append((left_region, right_region, partition))

        return results

    def random_horizontally_partition(self, region, num_partitions):
        results = []  # list of tuples of (top_region, bottom_region, parent partition)
        assert num_partitions > 0
        total_possible_partitions = region.x_2 - region.x_1 - 1
        if total_possible_partitions > 0:
            num_partitions = min(num_partitions, total_possible_partitions)
            for i in np.random.choice(total_possible_partitions, num_partitions):
                i = int(i + 1)
                # create two new regions
                top_region = self.scope_factory.get_region_instance(region.x_1,
                                                                    region.x_1 + i,
                                                                    region.y_1,
                                                                    region.y_2,
                                                                    region.image_n_rows,
                                                                    region.image_n_cols)

                bottom_region = self.scope_factory.get_region_instance(region.x_1 + i,
                                                                       region.x_2,
                                                                       region.y_1,
                                                                       region.y_2,
                                                                       region.image_n_rows,
                                                                       region.image_n_cols)

                partition = self.partition_factory.get_instance(children_scopes={top_region, bottom_region},
                                                                parent_scope=region)

                # sanity checking
                assert next(iter(partition.parents)).id == region.id

                results.append((top_region, bottom_region, partition))

        return results

    def is_smaller_than_vae(self, region):
        if (region.height, region.width) == (self.n_rows_fg, self.n_cols_fg):
            return False
        return region.height <= self.n_rows_fg and region.width <= self.n_cols_fg

    def create_region_graph(self, stride, n_rows_fg, n_cols_fg):
        region = self.root
        self.scopes.add(region)  # region_graph also keeps track of which scope nodes have been constructed

        # initialise queue
        regions_to_process = deque()
        regions_to_process.append(region)

        while regions_to_process:
            # get a region to process
            current_region = regions_to_process.popleft()

            if current_region.width == self.n_cols_fg and current_region.height == self.n_rows_fg:
                regions_to_consider = []

            # get all possible decompositions horizontally and vertically
            elif self.is_smaller_than_vae(current_region) and self.univariate_bg:
                vertical_region_splits = self.random_vertically_partition(current_region, 1)
                horizontal_region_splits = self.random_horizontally_partition(current_region, 1)
                regions_to_consider = vertical_region_splits + horizontal_region_splits
            else:
                vertical_region_splits = self.vertically_partition(current_region, stride, n_cols_fg)
                horizontal_region_splits = self.horizontally_partition(current_region, stride, n_rows_fg)
                regions_to_consider = vertical_region_splits + horizontal_region_splits

            for region_1, region_2, partition in regions_to_consider:
                # sanity check: constructed partition should always be new
                assert not partition in self.partitions, partition.vars
                self.add_partition(partition)

                self.sanity_check_scope(region_1)
                self.sanity_check_scope(region_2)

                # if freshly constructed, add to queue
                if not region_1 in self.scopes:
                    self.scopes.add(region_1)
                    regions_to_process.append(region_1)
                if not region_2 in self.scopes:
                    self.scopes.add(region_2)
                    regions_to_process.append(region_2)
