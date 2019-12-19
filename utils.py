import sys
import numpy as np
sys.path.append("LogisticCircuit")
sys.path.append("pypsdd")
from structure.Vtree import Vtree as LC_Vtree

import pdb

from vtree import Vtree as PSDD_Vtree
from manager import PSddManager
import psdd_io
from data import Inst


import itertools
from algo.LogisticCircuit import LogisticCircuit

import circuit_expect
from sympy import *

from scipy.special import expit

from EVCache import EVCache

'''
obsX[i] = -1 if unobserved, otherwise its the observation
'''


def brute_force_expectation(psdd, lgc, n, k=1, compute_prob=False, obsX=None):
    sum = np.float64(0.0)
    sum_all_prob = np.float64(0.0)
    run = 0

    if obsX is None:
        obsX = np.array([-1 for i in range(n)])

    obs_count = np.sum(obsX != -1)
    for x in itertools.product([0, 1], repeat=n - obs_count):
        X = np.copy(obsX)
        used = 0
        for i in range(n):
            if X[i] == -1:
                X[i] = x[used]
                used += 1

        run += 1
        if run % 1000 == 0:
            print("RUN ", run)
        inp = Inst.from_list(X, n, zero_indexed=True)
        lgc_features = lgc.calculate_features(np.array([X]))

        f = np.dot(lgc_features, lgc._parameters.T)
        if compute_prob:
            f = 1.0 / (1.0 + np.exp(-f))
        else:
            f = f**k
        p = psdd.pr_model(inp)
        print(X, " --> ", p)
        sum_all_prob += p
        sum += p * f

    ans = sum / sum_all_prob
    return ans[0]


def print_psdd(psdd):
    from collections import deque
    A = set()
    q = deque()
    q.appendleft(psdd)
    while q:
        now = q.pop()

        if now not in A:
            print(now)
            if not isinstance(now, tuple):
                print(now.theta)
            A.add(now)

        if isinstance(now, tuple):
            prime, sub = now
            q.appendleft(prime)
            q.appendleft(sub)
        elif now.is_decomposition():
            for e in now.elements:
                q.appendleft(e)


def predict_batch(psdd, lgc, X_test, T=None, brute_force=False, n=None, prob=True, batch_size=1000, is_regression=False):
    if not brute_force and T is None:
        raise Exception("Specify T when using taylor approx")

    if is_regression:
        prob = True
        T = 0

    N = X_test.shape[0]
    yHat = np.zeros((T + 1, X_test.shape[0], lgc._num_classes))
    for i in range(0, X_test.shape[0], batch_size):
        L = i
        R = min(i + batch_size, N)
        print("Doing batch [{}:{}]".format(L, R))
        obsX = X_test[L:R]
        cache = EVCache()
        exps = circuit_expect.Expectation(psdd, lgc, cache, obsX)

        if not is_regression:
            if brute_force:
                MEG = brute_force_expectation(psdd, lgc, n, compute_prob=True, obsX=obsX)
            else:
                MEG = sympy_taylor_aprox(psdd, lgc, cache, T, exps, obsX=obsX)

            yHat[:, L:R, :] = np.copy(MEG)

        else:
            yHat[:, L:R, :] = np.copy(exps)

    if prob:
        return yHat
    else:
        return np.argmax(yHat, axis=2)


"""
n variables
T number taylor expansion
"""


def sympy_taylor_aprox(psdd, lgc, cache, T, Alpha, obsX, extra_bias=False, print_debug=False):
    assert(Alpha.shape[0] == obsX.shape[0])

    classes = lgc._num_classes
    NN = obsX.shape[0]
    results = np.zeros((T + 1, NN, classes), dtype='float')

    x = symbols('x')
    sigmoid = 1.0 / (1.0 + exp(-x))

    #f0 = lambdify(x, sigmoid, 'numpy')
    results[0, :, :] = expit(Alpha)
    # for xi in range(NN):
    #     for c in range(classes):
    #         results[0][xi][c] = sigmoid.subs(x, Alpha[xi][c]).n()

    nFactor = 1.0
    for i in range(1, T + 1):
        fi = diff(sigmoid, x, i)

        li = lambdify([x], fi, ["numpy"])
        coeff_i_c = li(Alpha).astype('float')

        # lambda_i = np.vectorize(lambda z: fi.subs(x, z).n())
        # coeff_i_c = lambda_i(Alpha).astype('float')

        # coeff_i_c = np.zeros( Alpha.shape, dtype='float')
        # for xi in range(NN):
        #     for c in range(classes):
        #         coeff_i_c[xi][c] = fi.subs(x, Alpha[xi][c]).n()

        value = np.copy(results[i - 1, :, :])
        nFactor /= i

        momentI = np.zeros(Alpha.shape, dtype='float')

        for j in range(i + 1):
            momentI += circuit_expect.choose(i, j) * (-1)**(i - j) * \
                circuit_expect.moment(psdd, lgc, j, cache, obsX=obsX) * (Alpha)**(i - j)

        temp = momentI * coeff_i_c * nFactor
        value += temp
        results[i, :, :] = value

    return results


def exp_sum_log_choose_alpha(i, j, alpha):
    # returns numberical stable number for (|alpha|^(i-j) * choose(i, j)) / (i!)
    if abs(alpha - 0.0) < 1e-5:
        if i == j:
            log_val = 0.0
        else:
            log_val = -1000
    else:
        log_val = (i - j) * np.log(abs(alpha))

    for z in range(1, j + 1):
        log_val -= np.log(z)
    for z in range(1, i - j + 1):
        log_val -= np.log(z)

    val_sign = (-1)**(i - j)
    if alpha < 0:
        val_sign *= (-1) ** (i - j)

    temp = val_sign * exp(log_val)
    return temp
