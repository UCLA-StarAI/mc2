# Taking Expectation of a Logistic Circuit (without the logsitic) over a PSDD
import sys 
sys.path.append("LogisticCircuit")
sys.path.append("pypsdd")

from data import Inst

import numpy as np
from scipy.special import comb

from algo.LogisticCircuit import LogisticCircuit
from structure.CircuitNode import CircuitNode as LogisticCircuitNode
from structure.AndGate import AndGate as LogisticAndGate
from structure.CircuitNode import OrGate as LogisticOrGate
from structure.CircuitNode import CircuitTerminal as LogisticCircuitTerminal

from structure.CircuitNode import LITERAL_IS_TRUE, LITERAL_IS_FALSE

from psdd import PSddNode
from EVCache import EVCache

import pdb

def psdd_index(psdd):
    return psdd # TODO

def lgc_index(lgc):
    return lgc

N_comb = 31
COMB = np.zeros((N_comb, N_comb), dtype='float')
for i in range(N_comb):
    for j in range(N_comb):
        COMB[i][j] = comb(i, j, exact=False)

def choose(n , m):
    return COMB[n][m]
    # return np.float64(1.0) * comb(n, m, exact=True)


def Expectation(psdd, lgc, cache, obsX = None):
    """
    Main function to call

    lgc: The Logistic Circuit Node (from the circuit without the logistic in the root)
    psdd: A psdd node (which means it was OR) or tuple of psdd nodes (which means in was AND)
    cache: Cache of intermediary expectations
    """    
    if obsX is None:
        obsX = -1 * np.ones(lgc._num_variables)
        p_observed = np.float64(1.0)
    else:
        p_observed = np.ones( (obsX.shape[0], 1), dtype='float')
        for i in range(obsX.shape[0]):
            inp = Inst.from_list(obsX[i], lgc._num_variables, zero_indexed=True)
            p_observed[i,:] = psdd.probability(inp)

    if isinstance(lgc._root, LogisticAndGate):
        value = exp_g_AND(psdd, lgc._root, cache, obsX)
    elif isinstance(lgc._root, LogisticOrGate):
        value = exp_g_OR(psdd, lgc._root, cache, obsX)
    else:
        raise Exception("Logistic Circuit with no AND or OR gates in the root should not happen")
    
    value /= p_observed
    value += np.float64(lgc._bias)
    return np.copy(value)

def exp_g_AND(psdd, lgc: LogisticCircuitNode, cache: EVCache, obsX):
    prime, sub = psdd
    if not isinstance(prime, PSddNode) or not isinstance(sub,PSddNode):
        raise Exception("Psdd: Children of And gates should be Or gates")

    if not isinstance(lgc, LogisticAndGate):
        raise Exception("Lgc: Wrong Node type, should be AND")
    
    cached_value = cache.get_g( psdd_index(psdd), lgc_index(lgc))
    if cached_value is not None:
        return cached_value

    value = exp_g_OR(prime, lgc.prime, cache, obsX) + exp_g_OR(sub, lgc.sub, cache, obsX)    
    cache.put_g(psdd_index(psdd), lgc_index(lgc), value)
    return np.copy(value)


def exp_fg_AND(psdd, lgc: LogisticCircuitNode, cache: EVCache, obsX):
    #pdb.set_trace()
    prime, sub = psdd
    cached_value = cache.get_fg( psdd_index(psdd), lgc_index(lgc))
    if cached_value is not None:
        return cached_value

    value = np.float64(0.0)
    if isinstance(lgc, LogisticCircuitTerminal):
        raise("should not happen, and in logistic circuit being terminal")
    elif isinstance(lgc.prime, LogisticCircuitTerminal) and isinstance(lgc.sub, LogisticCircuitTerminal): 
        value = lgc.prime.parameter.reshape(1, -1) * exp_f_OR(sub, lgc.sub, cache, obsX) * exp_f_OR(prime, lgc.prime, cache, obsX) \
            + lgc.sub.parameter.reshape(1, -1) * exp_f_OR(sub, lgc.sub, cache, obsX) * exp_f_OR(prime, lgc.prime, cache, obsX)
    elif isinstance(lgc.prime, LogisticCircuitTerminal):
        value = exp_f_OR(prime, lgc.prime, cache, obsX) * exp_g_OR(sub, lgc.sub, cache, obsX) \
            + lgc.prime.parameter.reshape(1, -1) * exp_f_OR(sub, lgc.sub, cache, obsX) * exp_f_OR(prime, lgc.prime, cache, obsX)
    elif isinstance(lgc.sub, LogisticCircuitTerminal):
        value = exp_f_OR(sub, lgc.sub, cache, obsX) * exp_g_OR(prime, lgc.prime, cache, obsX) \
            + lgc.sub.parameter.reshape(1, -1) * exp_f_OR(sub, lgc.sub, cache, obsX) * exp_f_OR(prime, lgc.prime, cache, obsX)
    elif not prime.is_decomposition() or not sub.is_decomposition():
        raise Exception("This should not be possible: LGC non-terminal node, but psdd having a terminal node")
    else:
        value = exp_f_OR(prime, lgc.prime, cache, obsX) * exp_g_OR(sub, lgc.sub, cache, obsX) \
            + exp_f_OR(sub, lgc.sub, cache, obsX) * exp_g_OR(prime, lgc.prime, cache, obsX)

    cache.put_fg( psdd_index(psdd), lgc_index(lgc), value )
    return np.copy(value)

def exp_f_AND(psdd, lgc: LogisticCircuitNode, cache: EVCache, obsX):
    value = np.float64(0.0)
    if isinstance(lgc, LogisticCircuitTerminal):
        raise Exception("this might not happen at all")
    else:
        prime, sub = psdd
        cached_value = cache.get_f( psdd_index(psdd), lgc_index(lgc) )
        if cached_value is not None:
            return cached_value
        
        value = exp_f_OR(prime, lgc.prime, cache, obsX) * exp_f_OR(sub, lgc.sub, cache, obsX)
        cache.put_f(psdd_index(psdd), lgc_index(lgc), value)

    return np.copy(value)

def exp_g_OR(psdd: PSddNode, lgc: LogisticCircuitNode, cache: EVCache, obsX):
    #pdb.set_trace()
    cached_value = cache.get_g(psdd_index(psdd), lgc_index(lgc))
    if cached_value is not None:
        return cached_value

    value = np.float64(0.0)
    if isinstance(lgc, LogisticCircuitTerminal):
        value = np.float64(0.0)
    elif not psdd.is_decomposition():
        raise Exception("does it even go here")           
    else:
        temp_fg = np.float64(0.0)
        temp_f  = np.float64(0.0)
        for j in psdd.elements:
            temp_fg_k = np.float64(0.0)  # For the Red Sum from the notes
            temp_f_k = 0.0   # For the Blue Sum from the notes
            for k in lgc.elements:                
                temp_fg_k += exp_fg_AND(j, k, cache, obsX)

                PHI_K = k.parameter
                temp_f_k += PHI_K * exp_f_AND(j, k, cache, obsX)
            
            THETA_J = psdd.theta[j]
            temp_fg += THETA_J * temp_fg_k 
            temp_fg += THETA_J * temp_f_k

        value = temp_fg + temp_f

    cache.put_g(psdd_index(psdd), lgc_index(lgc), value)
    return np.copy(value)


# def exp_fg_OR(psdd: PSddNode, lgc: LogisticCircuitNode, cache: EVCache, obsX):
#     return exp_g_OR(psdd, lgc, cache, obsX)

"""
Given observation obsX does it agree/disagree with the leaf
always agree if that variable not observed 
"""
def agrees(psdd, lgc: LogisticCircuitTerminal, obsX):
    idx = lgc.var_index - 1
    if obsX[idx] == -1:
        return True
    
    if psdd.is_true():
        return obsX[idx] == lgc.var_value
    else:
        #return obsX[idx] == (psdd.literal > 0)
        if psdd.literal > 0:
            return obsX[idx] == 1
        else:
            return obsX[idx] == 0

def agrees_vectorized(psdd_isture, psdd_literal, lgc_var_index, lgc_var_val, obsX):
    idx = lgc_var_index - 1
    ans = np.zeros(obsX.shape[0], dtype='bool')
    for i in range(obsX.shape[0]):
        if obsX[i][idx] == -1:
            ans[i] = True
        elif psdd_isture:
            ans[i] = bool(obsX[i][idx] == lgc_var_val)
        elif psdd_literal > 0:
            ans[i] = bool(obsX[i][idx] == 1)
        else:
            ans[i] = bool(obsX[i][idx] == 0)
    return ans


def exp_f_OR(psdd: PSddNode, lgc: LogisticCircuitNode, cache: EVCache, obsX):
    cached_value = cache.get_f( psdd_index(psdd), lgc_index(lgc) )
    if cached_value is not None:
        return cached_value

    value = np.zeros((obsX.shape[0], 1), dtype='float')
    if isinstance(lgc, LogisticCircuitTerminal):
        # The psdd also has to be leaf node on the same variable        
        if psdd.is_false_sdd:
            raise Exception("unhandled for psdd.is_false_sdd")
        elif psdd.is_true():
            agrees_mask = agrees_vectorized(psdd.is_true(), psdd.literal, lgc._var_index, lgc._var_value, obsX)
            if lgc.var_value == LITERAL_IS_TRUE:# and agrees(psdd, lgc, obsX):
                value[agrees_mask, :] = psdd.theta[1]
            else:
                value[agrees_mask, :] = psdd.theta[0]

        elif not psdd.is_literal():
            print("lgc = [{}], psdd=[{}]".format(lgc, psdd))
            raise Exception("this probably should not happend, psdd non-literal but lgc is terminal node")
        else:
            agrees_mask = agrees_vectorized(psdd.is_true(), psdd.literal, lgc._var_index, lgc._var_value, obsX)
            if lgc.var_value == LITERAL_IS_TRUE and psdd.literal > 0:# and agrees(psdd, lgc, obsX):
                value[agrees_mask, :] = psdd.theta[1]
            elif lgc.var_value == LITERAL_IS_FALSE and psdd.literal < 0:# and agrees(psdd, lgc, obsX):
                value[agrees_mask, :] = psdd.theta[0]
            else:
                pass

    elif not psdd.is_decomposition():
        raise Exception("Should not happen: Psdd leaf node but LGC not leaf node")
    else:
        for j in psdd.elements:
            for k in lgc.elements:    
                value += exp_f_AND(j, k, cache, obsX) * psdd.theta[j]

    cache.put_f(psdd_index(psdd), lgc_index(lgc), value)
    return np.copy(value)

############################################################################

def moment(psdd, lgc, moment, cache, obsX = None, extraBias = None):
    value = np.longdouble(0.0)
    if obsX is None:
        obsX = -1 * np.ones(lgc._num_variables)
        p_observed = np.float64(1.0)
    else:
        p_observed = np.zeros( (obsX.shape[0], 1), dtype='float')
        for i in range(obsX.shape[0]):
            inp = Inst.from_list(obsX[i], lgc._num_variables, zero_indexed=True)
            p_observed[i,:] = psdd.probability(inp)

    BIAS = np.copy(np.longdouble(lgc._bias))
    if extraBias is not None:
        BIAS += extraBias

    for z in range(0, moment + 1):
        if isinstance(lgc._root, LogisticAndGate):
            temp = choose(moment, z) * (BIAS**z) * moment_g_AND(psdd, lgc._root, moment-z, cache, obsX)
            if z == moment:
                # to cancel the effect of dividing bias**moment by p_observed
                temp *= p_observed 
            value += temp
        elif isinstance(lgc._root, LogisticOrGate):            
            temp = choose(moment, z) * (BIAS**z) * moment_g_OR(psdd, lgc._root,  moment-z, cache, obsX)
            if z == moment:
                # to cancel the effect of dividing bias**moment by p_observed
                temp *= p_observed
            value += temp
        else:
            raise Exception("Logistic Circuit with no AND or OR gates in the root should not happen")
    value /= p_observed
    return np.copy(value)

def moment_g_AND(psdd, lgc: LogisticCircuitNode, moment, cache: EVCache, obsX):
    if moment == 0:
        return np.ones( (obsX.shape[0], 1), dtype='float') # np.float64(1.0)
    
    prime, sub = psdd
    cached_value = cache.get_moment_g( psdd_index(psdd), lgc_index(lgc) , moment)
    if cached_value is not None:
        return cached_value

    value = np.zeros( (obsX.shape[0], 1), dtype='float') # np.float64(0.0)
    for z in range(0, moment+1):
        A = moment_g_OR(prime, lgc.prime, z, cache, obsX) 
        B = moment_g_OR(sub, lgc.sub, moment - z, cache, obsX)
        value += choose(moment, z) * A * B

    cache.put_moment_g( psdd_index(psdd), lgc_index(lgc) , moment, value)
    return np.copy(value)

def moment_fg_AND(psdd, lgc: LogisticCircuitNode, moment, cache: EVCache, obsX):
    if moment == 0:
        return exp_f_AND(psdd, lgc, cache, obsX)

    prime, sub = psdd
    cached_value = cache.get_moment_fg( psdd_index(psdd), lgc_index(lgc) , moment)
    if cached_value is not None:
        return cached_value
    
    value = np.zeros( (obsX.shape[0], 1), dtype='float') # np.float64(0.0)
    for z in range(0 , moment + 1):
        if isinstance(lgc, LogisticCircuitTerminal):
            raise("should not happen, and in logistic circuit being terminal")

        if isinstance(lgc.prime, LogisticCircuitTerminal):
            A = (lgc.prime.parameter)**z * exp_f_OR(prime, lgc.prime, cache, obsX) 
        else:
            A = moment_fg_OR(prime, lgc.prime, z, cache, obsX)

        if isinstance(lgc.sub, LogisticCircuitTerminal):
            B = (lgc.sub.parameter)**(moment-z) *  exp_f_OR(sub, lgc.sub, cache, obsX)
        else:
            B = moment_fg_OR(sub, lgc.sub, moment - z, cache, obsX)

        value = value + choose(moment, z) *  A * B

    cache.put_moment_fg( psdd_index(psdd), lgc_index(lgc) , moment, value)
    return np.copy(value)


def moment_g_OR(psdd: PSddNode, lgc: LogisticCircuitNode, moment, cache: EVCache, obsX):
    if moment == 0:
        return np.ones( (obsX.shape[0], 1), dtype='float') #np.float64(1.0)
    # if moment == 1:
    #     return exp_g_OR(psdd, lgc, cache, obsX)

    cached_value = cache.get_moment_g(psdd_index(psdd), lgc_index(lgc), moment)
    if cached_value is not None:
        return cached_value

    if isinstance(lgc, LogisticCircuitTerminal):
        value = np.zeros( (obsX.shape[0], 1), dtype='float') #np.float64(0.0)
    elif not psdd.is_decomposition():
        raise Exception("should not go here, unhandled")
    else:      
        value = np.zeros( (obsX.shape[0], 1), dtype='float') # np.float64(0.0)
        for j in psdd.elements:
            temp_j_sum = np.float64(0.0)
            for k in lgc.elements:
                for z in range(0, moment+1):
                    #A = psdd.theta[j]
                    B = (k.parameter)**(moment - z)           
                    C = moment_fg_AND(j, k, z, cache, obsX)

                    temp_j_sum += choose(moment, z) * B * C

            value = value + psdd.theta[j] * temp_j_sum

    cache.put_moment_g(psdd_index(psdd), lgc_index(lgc), moment, value)
    return np.copy(value)

def moment_fg_OR(psdd: PSddNode, lgc: LogisticCircuitNode, moment, cache: EVCache, obsX):
    if moment == 0:
        return exp_f_OR(psdd, lgc, cache, obsX)
    return moment_g_OR(psdd, lgc, moment, cache, obsX)


def forward_comp_exp(cache: EVCache, obsX):        
    cache.f_cache.clear()
    cache.g_cache.clear()
    cache.fg_cache.clear()
    for computation in cache.exp_order[::-1]:
        if computation[0] == "f":
            if isinstance(computation[2], LogisticAndGate):
                exp_f_AND(computation[1], computation[2], cache, obsX)
            else:
                exp_f_OR(computation[1], computation[2], cache, obsX) 
        elif computation[0] == "g":
            if isinstance(computation[2], LogisticAndGate):
                exp_g_AND(computation[1], computation[2], cache, obsX)
            else:
                exp_g_OR(computation[1], computation[2], cache, obsX) 
        else:
            if isinstance(computation[2], LogisticAndGate):
                exp_fg_AND(computation[1], computation[2], cache, obsX)
            else:
                exp_g_OR(computation[1], computation[2], cache, obsX) 

        