from collections import deque, defaultdict
from abc import ABC, abstractmethod

VTREE_FORMAT = """c ids of vtree nodes start at 0
c ids of variables start at 1
c vtree nodes appear bottom-up, children before parents
c
c file syntax:
c vtree number-of-nodes-in-vtree
c L id-of-leaf-vtree-node id-of-variable
c I id-of-internal-vtree-node id-of-left-child id-of-right-child
c
"""


class Vtree(ABC):

    def __init__(self, index):
        self._index = index

    @property
    def index(self):
        return self._index

    @property
    def var_count(self):
        return self._var_count

    @abstractmethod
    def is_leaf(self):
        pass

    def bfs(self):
        visited = []
        nodes_to_visit = deque()
        nodes_to_visit.append(self)
        while nodes_to_visit:
            n = nodes_to_visit.popleft()
            visited.append(n)
            if not n.is_leaf():
                nodes_to_visit.append(n.left)
                nodes_to_visit.append(n.right)

        return visited

    @staticmethod
    def read(file):
        with open(file, 'r') as vtree_file:
            line = 'c'
            while line[0] == 'c':
                line = vtree_file.readline()
            if line.strip().split(' ')[0] != 'vtree':
                raise ValueError('Number of vtree nodes is not specified')
            num_nodes = int(line.strip().split(' ')[1])
            nodes = [None] * num_nodes
            root = None
            for line in vtree_file.readlines():
                line_as_list = line.strip().split(' ')
                if line_as_list[0] == 'L':
                    root = VtreeLeaf(int(line_as_list[1]), int(line_as_list[2]))
                    nodes[int(line_as_list[1])] = root
                elif line_as_list[0] == 'I':
                    root = VtreeIntermediate(int(line_as_list[1]),
                                             nodes[int(line_as_list[2])], nodes[int(line_as_list[3])])
                    nodes[int(line_as_list[1])] = root
                else:
                    raise ValueError('Vtree node could only be L or I')
            return root

    def save(self, file):

        leaves_before_parents = list(reversed(self.bfs()))
        n_nodes = len(leaves_before_parents)
        print('There are ', n_nodes)
        with open(file, 'w') as f_out:
            f_out.write(VTREE_FORMAT)
            f_out.write(f'vtree {n_nodes}\n')

            for n in leaves_before_parents:
                if isinstance(n, VtreeLeaf):
                    f_out.write(f'L {n.index} {n.var}\n')
                elif isinstance(n, VtreeIntermediate):
                    f_out.write(f'I {n.index} {n.left.index} {n.right.index}\n')
                else:
                    raise ValueError('Unrecognized vtree node type', n)


class VtreeLeaf(Vtree):

    def __init__(self, index, variable):
        super(VtreeLeaf, self).__init__(index)
        self._var = variable
        self._var_count = 1

    def is_leaf(self):
        return True

    @property
    def var(self):
        return self._var

    @property
    def variables(self):
        return set([self._var])


class VtreeIntermediate(Vtree):

    def __init__(self, index, left, right):
        super(VtreeIntermediate, self).__init__(index)
        self._left = left
        self._right = right
        self._variables = set()
        self._var_count = self._left.var_count + self._right.var_count
        self._variables.update(self._left.variables)
        self._variables.update(self._right.variables)

    def is_leaf(self):
        return False

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def variables(self):
        return self._variables

    @property
    def var(self):
        return 0


#
# Generate vtrees
#
import numpy as np
RAND_SEED = 1337


def balanced_random_split(variables, index, rand_gen):

    n_vars = len(variables)

    if n_vars > 1:
        rand_gen.shuffle(variables)
        mid = n_vars // 2
        var_left, var_right = variables[:mid], variables[mid:]

        node_left, id_left = balanced_random_split(var_left, index, rand_gen)
        node_right, id_right = balanced_random_split(var_right, id_left, rand_gen)

        v = VtreeIntermediate(id_right, node_left, node_right)
        return v, id_right + 1
    else:

        v = VtreeLeaf(index, variables[0])
        return v, index + 1


def unbalanced_random_split(variables, index, rand_gen, beta_prior=(0.3, 0.3)):

    n_vars = len(variables)

    if n_vars > 1:
        rand_gen.shuffle(variables)

        rand_split = rand_gen.beta(a=beta_prior[0], b=beta_prior[1])
        mid = max(min(int(rand_split * n_vars), n_vars - 1), 1)
        var_left, var_right = variables[:mid], variables[mid:]

        node_left, id_left = unbalanced_random_split(var_left, index, rand_gen, beta_prior)
        node_right, id_right = unbalanced_random_split(var_right, id_left, rand_gen, beta_prior)

        v = VtreeIntermediate(id_right, node_left, node_right)
        return v, id_right + 1
    else:

        v = VtreeLeaf(index, variables[0])
        return v, index + 1


def generate_random_vtree(n_vars, rand_gen=None, balanced=True, beta_prior=(0.3, 0.3)):

    if rand_gen is None:
        rand_gen = np.random.RandomState(RAND_SEED)

    vars = np.arange(n_vars) + 1

    if balanced:
        v, _ = balanced_random_split(vars, index=0, rand_gen=rand_gen)
    else:
        v, _ = unbalanced_random_split(vars, index=0, rand_gen=rand_gen, beta_prior=beta_prior)

    return v
