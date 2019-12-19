import sys
import numpy as np
sys.path.append("LogisticCircuit")
sys.path.append("pypsdd")
sys.path.append(".")
from structure.Vtree import Vtree as LC_Vtree

from collections import defaultdict

import pdb

from vtree import Vtree as PSDD_Vtree
from manager import PSddManager
import psdd_io
from data import Inst, InstMap


import itertools
from algo.LogisticCircuit import LogisticCircuit

import circuit_expect
from sympy import *
from utils import *

from scipy.special import logit
from scipy.special import expit

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


def taylor_aprox(psdd, lgc, n):
    sum = 0.5
    coeff = [1 / 4.0, -1 / 48.0, 1 / 480.0, -17 / 80640.0, 31 / 1451520.0, -691 / 319334400.0,
             5461 / 24908083200.0, -929569 / 41845579776000.0]#, 3202291 /
             #1422749712384000.0, -221930581 / 973160803270656000.0,
             #4722116521 / 204363768686837760000.0]
    for k, c in enumerate(coeff):
        cur = brute_force_expectation(psdd, lgc, n, k=2 * k + 1, compute_prob=False)
        sum += c * cur
    return sum



if __name__ == '__main__':
    print("Loading Logistic Circuit..")

    VTREE_FILE = "test/circuits/4.vtree"
    GLC_FILE = "test/circuits/4.glc"
    PSDD_FILE = "test/circuits/4.psdd"
    CLASSES = 2
    N = 4

    # FOLDER = "notebooks/rand-gen-grid/exp-D9-N500-C6-B1/D9-N500-C6-B1"
    # VTREE_FILE = FOLDER + ".vtree"
    # GLC_FILE = FOLDER + ".glc"
    # PSDD_FILE = FOLDER + ".psdd"
    # CLASSES = 6
    # N = 9

    # VTREE_FILE = "test/circuits/5.vtree"
    # GLC_FILE = "test/circuits/5.glc"
    # PSDD_FILE = "test/circuits/5.psdd"
    # CLASSES = 2
    # N = 2

    # VTREE_FILE = "notebooks/exp-D15-N1000-C4-balanced.vtree"
    # GLC_FILE = "notebooks/exp-D15-N1000-C4.glc"
    # PSDD_FILE = "notebooks/exp-D15-N1000-C4.psdd"
    # CLASSES = 4
    # N = 15

    # VTREE_FILE = "exp/test-adult/adult-test-I/adult.vtree"
    # GLC_FILE   = "exp/test-adult/adult-test-I/adult.glc"
    # PSDD_FILE  = "exp/test-adult/adult-test-I/adult.psdd"
    # CLASSES = 2
    # N = 157

    
    # VTREE_FILE = "exp/new-reg-circuit-grid/elevators/elevators_20190520-185859/elevators.vtree"
    # GLC_FILE   = "exp/new-reg-circuit-grid/elevators/elevators_20190520-185859/best/elevators.glc"
    # PSDD_FILE  = "exp/new-reg-circuit-grid/elevators/elevators_20190520-185859/best/elevators.psdd"
    # CLASSES = 1
    # N = 182

    # VTREE_FILE = "exp/mnist-final/mnist.vtree"
    # GLC_FILE = "exp/mnist-final/mnist.glc"
    # PSDD_FILE = "exp/mnist-final/mnist.psdd"
    # CLASSES = 10
    # N = 28*28
    
    lc_vtree = LC_Vtree.read(VTREE_FILE)
    with open(GLC_FILE) as circuit_file:
        lgc = LogisticCircuit(lc_vtree, CLASSES, circuit_file=circuit_file)
        
    print("Loading PSDD..")
    psdd_vtree = PSDD_Vtree.read(VTREE_FILE)
    manager = PSddManager(psdd_vtree)
    psdd = psdd_io.psdd_yitao_read(PSDD_FILE, manager)
    #################

    try:
        from time import perf_counter
    except:
        from time import time
        perf_counter = time

    X = np.zeros( (1, N) ) 

    start_t = perf_counter()
    cache = EVCache()
    ans = circuit_expect.Expectation(psdd, lgc, cache, X)
    end_t = perf_counter()
    4
    print("Time taken {}".format(end_t - start_t))