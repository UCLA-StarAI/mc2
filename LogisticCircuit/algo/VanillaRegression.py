import copy
import gc
from collections import deque
import logging
from time import perf_counter

import numpy as np
from sklearn.metrics import mean_squared_error

# from algo.Ridge import Ridge
from sklearn.linear_model import Ridge
from structure.AndGate import AndGate
from structure.CircuitNode import CircuitNode, OrGate, CircuitTerminal
from structure.CircuitNode import LITERAL_IS_TRUE, LITERAL_IS_FALSE, LITERAL_IS_TAUTOLOGY
from structure.Vtree import Vtree

FORMAT = """c variables (from inputs) start from 1
c ids of logistic circuit nodes start from 0
c nodes appear bottom-up, children before parents
c the last line of the file records the bias parameter
c three types of nodes:
c	T (terminal nodes that correspond to true literals)
c	F (terminal nodes that correspond to false literals)
c   S (terminal nodes that correspond to tautology literals)
c	D (OR gates)
c
c file syntax:
c Regression Circuit
c T id-of-true-literal-node id-of-vtree variable parameters
c F id-of-false-literal-node id-of-vtree variable parameters
c D id-of-or-gate id-of-vtree number-of-elements (id-of-prime id-of-sub parameters)s
c B bias-parameters
c
"""


class RegressionCircuit(object):
    def __init__(self, vtree, circuit_file=None):
        self._vtree = vtree
        self._largest_index = 0
        self._num_variables = vtree.var_count

        self._precreated_terminal_nodes = [None] * 3 * self._num_variables
        self._terminal_nodes = None
        self._elements = None
        self._parameters = None
        self._bias = np.random.random_sample(size=1)

        if circuit_file is None:
            self._generate_all_terminal_nodes(vtree)
            self._root = self._new_regression_psdd(vtree)
        else:
            self._root = self.load(circuit_file)

        self._serialize()

    @property
    def vtree(self):
        return self._vtree

    @property
    def num_parameters(self):
        return self._parameters.size

    @property
    def parameters(self):
        return self._parameters

    def _generate_all_terminal_nodes(self, vtree: Vtree):
        if vtree.is_leaf():
            var_index = vtree.var
            self._precreated_terminal_nodes[var_index - 1] = CircuitTerminal(
                self._largest_index,
                vtree,
                var_index,
                LITERAL_IS_TAUTOLOGY,
                np.random.random_sample(size=(self._num_classes,)),
            )
            self._largest_index += 1
            self._precreated_terminal_nodes[self._num_variables + var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_TRUE, np.random.random_sample(size=1)
            )
            self._largest_index += 1
            self._precreated_terminal_nodes[2 * self._num_variables + var_index - 1] = CircuitTerminal(
                self._largest_index, vtree, var_index, LITERAL_IS_FALSE, np.random.random_sample(size=1)
            )
            self._largest_index += 1
        else:
            self._generate_all_terminal_nodes(vtree.left)
            self._generate_all_terminal_nodes(vtree.right)

    def _new_regression_psdd(self, vtree) -> CircuitNode:
        left_vtree = vtree.left
        right_vtree = vtree.right
        prime_variable = left_vtree.var
        sub_variable = right_vtree.var
        if left_vtree.is_leaf():
            left_node = self._precreated_terminal_nodes[prime_variable - 1]
        else:
            left_node = self._new_logistic_psdd(left_vtree)
        if right_vtree.is_leaf():
            right_node = self._precreated_terminal_nodes[sub_variable - 1]
        else:
            right_node = self._new_logistic_psdd(right_vtree)
        elements = [AndGate(left_node, right_node, np.random.random_sample(size=(self._num_classes,)))]
        elements[0].splittable_variables = copy.deepcopy(vtree.variables)
        root = OrGate(self._largest_index, vtree, elements)
        self._largest_index += 1
        return root

    def _serialize(self):
        """Serialize all the decision nodes in the logistic psdd.
           Serialize all the elements in the logistic psdd. """
        self._decision_nodes = [self._root]
        self._terminal_nodes = []
        self._elements = []
        decision_node_indices = set()
        terminal_node_indices = set()
        decision_node_indices.add(self._root.index)
        unvisited = deque()
        unvisited.append(self._root)
        while len(unvisited) > 0:
            current = unvisited.popleft()
            for element in current.elements:
                self._elements.append(element)
                element.flag = False
                if isinstance(element.prime, OrGate) and element.prime.index not in decision_node_indices:
                    decision_node_indices.add(element.prime.index)
                    self._decision_nodes.append(element.prime)
                    unvisited.append(element.prime)
                if isinstance(element.sub, OrGate) and element.sub.index not in decision_node_indices:
                    decision_node_indices.add(element.sub.index)
                    self._decision_nodes.append(element.sub)
                    unvisited.append(element.sub)
                if isinstance(element.prime, CircuitTerminal) and element.prime.index not in terminal_node_indices:
                    terminal_node_indices.add(element.prime.index)
                    self._terminal_nodes.append(element.prime)
                if isinstance(element.sub, CircuitTerminal) and element.sub.index not in terminal_node_indices:
                    terminal_node_indices.add(element.sub.index)
                    self._terminal_nodes.append(element.sub)
        self._parameters = self._bias.reshape(-1, 1)
        for terminal_node in self._terminal_nodes:
            # print(self._parameters.shape, terminal_node.parameter.reshape(-1, 1).shape)
            self._parameters = np.concatenate((self._parameters, terminal_node.parameter.reshape(-1, 1)), axis=1)
        for element in self._elements:
            self._parameters = np.concatenate((self._parameters, element.parameter.reshape(-1, 1)), axis=1)
        gc.collect()

    def _record_learned_parameters(self, parameters):
        self._parameters = copy.deepcopy(parameters).reshape(1, -1)
        print("todo fix the _record_learned_parameters")
        # print('PARAMS', self._parameters.shape)
        # self.bias = self._parameters[:, 0]
        # for i in range(len(self._terminal_nodes)):
        #     self._terminal_nodes[i].parameter = self._parameters[:, i + 1]
        # for i in range(len(self._elements)):
        #     self._elements[i].parameter = self._parameters[:, i + 1 + 2 * self._num_variables]
        gc.collect()

    def calculate_features(self, images: np.array):
        num_images = images.shape[0]
        for terminal_node in self._terminal_nodes:
            terminal_node.calculate_prob(images)
        for decision_node in reversed(self._decision_nodes):
            decision_node.calculate_prob()
        self._root.feature = np.ones(shape=(num_images,), dtype=np.float64)
        for decision_node in self._decision_nodes:
            decision_node.calculate_feature()
        # bias feature
        bias_features = np.ones(shape=(num_images,), dtype=np.float64)
        terminal_node_features = np.vstack([terminal_node.feature for terminal_node in self._terminal_nodes])
        element_features = np.vstack([element.feature for element in self._elements])
        features = np.vstack((bias_features, terminal_node_features, element_features))
        for terminal_node in self._terminal_nodes:
            terminal_node.feature = None
            terminal_node.prob = None
        for element in self._elements:
            element.feature = None
            element.prob = None
        return features.T

    # def calculate_accuracy(self, data):
    #     """Calculate accuracy given the learned parameters on the provided data."""
    #     y = self.predict(data.features)
    #     accuracy = np.sum(y == data.labels) / data.num_samples
    #     return accuracy

    def calculate_error(self, data):
        """Calculate accuracy given the learned parameters on the provided data."""
        y = self.predict(data.features)
        mse = mean_squared_error(data.labels, y)
        return mse

    # def predict(self, features):
    #     y = self.predict_prob(features)
    #     return np.argmax(y, axis=1)

    def predict(self, features):
        return np.dot(features, self._parameters.T)

    # def predict_prob(self, features):
    #     """Predict the given images by providing their corresponding features."""
    #     y = 1.0 / (1.0 + np.exp(-np.dot(features, self._parameters.T)))
    #     return y

    def learn_parameters(self, data, num_iterations, num_cores=-1, alpha=1.0, tol=0.001, solver="auto", rand_gen=None):
        """Logistic Psdd's parameter learning is reduced to logistic regression.
        We use mini-batch SGD to optimize the parameters."""

        model = Ridge(
            alpha=alpha,
            fit_intercept=False,
            normalize=False,
            copy_X=True,
            max_iter=num_iterations,
            tol=tol,
            solver=solver,
            # coef_=self._parameters,
            random_state=rand_gen,
        )

        # model = LogisticRegression(solver='saga', fit_intercept=False,
        #                            # multi_class='ovr',
        #                            max_iter=num_iterations, C=10.0, warm_start=True, tol=1e-5,
        #                            coef_=self._parameters, n_jobs=num_cores)
        model.fit(data.features, data.labels)
        # print('PARAMS', self._parameters.shape, model.coef_.shape)
        self._record_learned_parameters(model.coef_)
        gc.collect()

    def save(self, f):
        self._serialize()
        f.write(FORMAT)
        f.write(f"Regression Circuit\n")
        for terminal_node in self._terminal_nodes:
            terminal_node.save(f)
        for decision_node in reversed(self._decision_nodes):
            decision_node.save(f)
        f.write("B")
        for parameter in self._bias:
            f.write(f" {parameter}")
        f.write("\n")

    def load(self, f):
        # read the format at the beginning
        line = f.readline()
        while line[0] == "c":
            line = f.readline()

        # serialize the vtree
        vtree_nodes = dict()
        unvisited_vtree_nodes = deque()
        unvisited_vtree_nodes.append(self._vtree)
        while len(unvisited_vtree_nodes):
            node = unvisited_vtree_nodes.popleft()
            vtree_nodes[node.index] = node
            if not node.is_leaf():
                unvisited_vtree_nodes.append(node.left)
                unvisited_vtree_nodes.append(node.right)

        # extract the saved logistic circuit
        nodes = dict()
        line = f.readline()
        while line[0] == "T" or line[0] == "F" or line[0] == "S":
            line_as_list = line.strip().split(" ")
            literal_type, var = line_as_list[0], int(line_as_list[3])
            index, vtree_index = int(line_as_list[1]), int(line_as_list[2])
            parameters = []
            for i in range(self._num_classes):
                parameters.append(float(line_as_list[4 + i]))
            parameters = np.array(parameters, dtype=np.float64)
            if literal_type == "T":
                self._precreated_terminal_nodes[self._num_variables + var - 1].parameter = parameters
                nodes[index] = (self._precreated_terminal_nodes[self._num_variables + var - 1], {var})
            elif literal_type == "F":
                self._precreated_terminal_nodes[2 * self._num_variables + var - 1].parameter = parameters
                nodes[index] = (self._precreated_terminal_nodes[2 * self._num_variables + var - 1], {-var})
            else:
                self._precreated_terminal_nodes[var - 1].parameter = parameters
                nodes[index] = (self._precreated_terminal_nodes[var - 1], {-var})
            self._largest_index = max(self._largest_index, index)
            line = f.readline()
        self._terminal_nodes = [x[0] for x in nodes.values()]

        root = None
        while line[0] == "D":
            line_as_list = line.strip().split(" ")
            index, vtree_index, num_elements = int(line_as_list[1]), int(line_as_list[2]), int(line_as_list[3])
            elements = []
            variables = set()
            for i in range(num_elements):
                # prime_index = int(line_as_list[i * (self._num_classes + 2) + 4].strip('('))
                # sub_index = int(line_as_list[i * (self._num_classes + 2) + 5])
                #
                # FIXME: remove constants
                prime_index = int(line_as_list[i * (1 + 2) + 4].strip("("))
                sub_index = int(line_as_list[i * (1 + 2) + 5])
                element_variables = nodes[prime_index][1].union(nodes[sub_index][1])
                variables = variables.union(element_variables)
                splittable_variables = set()
                for variable in element_variables:
                    if -variable in element_variables:
                        splittable_variables.add(abs(variable))
                parameters = []
                # for j in range(self._num_classes):
                #     parameters.append(
                #         float(line_as_list[i * (self._num_classes + 2) + 6 + j].strip(')')))
                parameters.append(float(line_as_list[i * (1 + 2) + 6].strip(")")))
                parameters = np.array(parameters, dtype=np.float64)
                elements.append(AndGate(nodes[prime_index][0], nodes[sub_index][0], parameters))
                elements[-1].splittable_variables = splittable_variables
            nodes[index] = (OrGate(index, vtree_nodes[vtree_index], elements), variables)
            root = nodes[index][0]
            self._largest_index = max(self._largest_index, index)
            line = f.readline()

        if line[0] != "B":
            raise ValueError("The last line in a circuit file must record the bias parameters.")
        self._bias = np.array([float(x) for x in line.strip().split(" ")[1:]], dtype=np.float64)

        gc.collect()
        return root


def learn_regression_circuit(vtree, train, max_iter_sl=1000, max_iter_pl=1000, depth=20, num_splits=10, alpha=0.2):

    # #
    # # FIXEME: do we need this?
    # X[np.where(X == 0.0)[0]] = 1e-5
    # X[np.where(X == 1.0)[0]] -= 1e-5

    # train = Dataset(train_x, train_y)

    error_history = []

    circuit = RegressionCircuit(vtree)
    train.features = circuit.calculate_features(train.images)

    logging.info(f"The starting circuit has {circuit.num_parameters} parameters.")
    train.features = circuit.calculate_features(train.images)
    train_acc = circuit.calculate_error(train)
    logging.info(f" error: {train_acc:.5f}")
    error_history.append(train_acc)

    return circuit, error_history
