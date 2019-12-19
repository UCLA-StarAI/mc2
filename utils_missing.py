import matplotlib
matplotlib.use('Agg')

import math
from time import perf_counter
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
import pickle
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error

from utils import predict_batch

import statsmodels.imputation.mice as mice
import pandas as pd

from data import Inst, InstMap

from copy import deepcopy

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


def conditional_likelihood_k(P, Q, eps=1e-14):
    return (0.0 - np.sum(P * np.log(Q + eps))) / (1.0 * P.shape[0])


def predict_nbk_with_missing(X, NB, missing, prob=False):
    mX = X
    mX_ = 1 - X
    if not missing is None:
        mX = mX * (1 - missing)
        mX_ = mX_ * (1 - missing)

    mX = np.matrix(mX)
    mX_ = np.matrix(mX_)

    mA = NB.feature_log_prob_
    mA_ = np.log(1 - np.exp(mA))

    mA = np.matrix(mA).T
    mA_ = np.matrix(mA_).T

    mP = NB.class_log_prior_

    Z1 = np.exp(mX * mA + mX_ * mA_ + mP)
    Z2 = Z1 / np.sum(Z1, axis=1)
    if prob:
        return np.array(Z2)
    else:
        yHatz = np.argmax(Z2, axis=1)
        yHat = np.array([int(yHatz[i][0]) for i in range(yHatz.shape[0])]).reshape(1, -1)
        return yHat


def run_missing_exp(X_test, y_test, psdd, glc, setting):
    X_impute_median = np.median(X_test, axis=0)
    X_impute_mean = np.mean(X_test, axis=0)

    k_all = []
    missing_err_lr_median_all = []
    missing_err_lr_mean_all = []
    missing_err_sample_all = []
    # missing_err_lr_em_impute_all = []
    # missing_err_lr_mice_impute_all = []

    isRegresion = setting["regression"] if "regression" in setting else False
    percentage = setting["percentage"] if "percentage" in setting else False

    # do_emImpute = setting["emImpute"] if "emImpute" in setting else False
    # do_miceImpute = setting["miceImpute"] if "miceImpute" in setting else False

    do_sample = setting["sample_method"] if "sample_method" in setting else False
    # if do_sample:
    if "sample_size" in setting:
        sample_size = int(setting["sample_size"])
    else:
        sample_size = 11

    T = setting["T"] if "T" in setting else 1

    missing_err_circuit_all = []
    missing_all = dict()
    predictions = dict()  # (model, k, R)
    missing_err_circuit_all_t = [deepcopy([]) for it in range(T + 1)]

    function = setting["function"] if "function" in setting else None
    if function is None:
        if not isRegresion:
            function = accuracy_score
        else:
            def sqrt_mse(x, y): return np.sqrt(mean_squared_error(x, y))
            function = sqrt_mse

    print("Using following function: ")
    print(function)

    repeat = setting["repeat"] if "repeat" in setting else 1
    FEATURES = setting["features"] if "features" in setting else None
    if FEATURES is None:
        NNN = X_test.shape[1]
        FEATURES = np.array([i for i in range(NNN)])
    else:
        FEATURES = np.array(FEATURES)

    print("Possible features to remove: {}".format(FEATURES.shape[0]))
    K = setting["k"]

    for ki, k_percent in enumerate(K):
        if percentage:
            k = int((k_percent / 100.0) * FEATURES.shape[0])
        else:
            k = k_percent

        print("K = {}".format(k))

        if k > FEATURES.shape[0]:
            print("Early stop: Only had {} features possible to remove vs {}".format(FEATURES.shape[0], k))
            break

        cur_lr_median = []
        cur_lr_mean = []
        # cur_em_impute = []
        # cur_mice_impute = []
        cur_circuit = []
        cur_sample = []

        cur_circuit_t = [deepcopy([]) for it1 in range(T + 1)]
        # cur_circuit_t = dict()
        # for it1 in range(T+1):
        #     cur_circuit_t[it1] = []

        for R in range(repeat):
            if R % 10 == 0:
                print("\t R = {}".format(R))
            X_test_median = np.array(X_test, dtype='float')
            X_test_mean = np.array(X_test, dtype='float')
            X_test_sample = np.array(X_test, dtype='int')
            # X_test_em_impute = np.array(X_test, dtype='float')
            # X_test_mice_impute = np.array(X_test, dtype='float')
            X_test_circuit = np.array(X_test, dtype='float')
            missing = np.zeros(X_test.shape, dtype=bool)

            for i in range(X_test.shape[0]):
                miss = np.random.choice(FEATURES, k, replace=False)

                missing[i][miss] = True
                X_test_median[i][miss] = X_impute_median[miss]
                X_test_mean[i][miss] = X_impute_mean[miss]
                X_test_sample[i][miss] = -1
                # X_test_em_impute[i][miss] = np.nan
                # X_test_mice_impute[i][miss] = np.nan
                X_test_circuit[i][miss] = -1

            missing_all[(ki, R)] = np.copy(missing)

            # if do_emImpute:
            #     import time
            #     start = time.time()
            #     loops = 6
            #     print ("\tStarting to em impute with loops = {}".format(loops))
            #     X_test_em_impute = impyute.em(X_test_em_impute, loops = loops)
            #     end = time.time()
            #     print ("\tDone imputing! " + str( end - start ) )
            # else:
            #     X_test_em_impute = np.zeros(X_test.shape)

            # if do_miceImpute:
            #     import time
            #     start = time.time()
            #     print ("\tStarting to mice impute")
            #     X_test_mice_impute = impyute.mice(X_test_mice_impute)
            #     end = time.time()
            #     print ("\tDone imputing! " + str( end - start ) )
            # else:
            #      X_test_mice_impute = np.zeros(X_test.shape)

            if do_sample:   
                sample_start_t = perf_counter()
            
                sample_predict_all = [] 
                for x_ind in range(len(y_test)):
                    ins_inp = InstMap.from_list(X_test_sample[x_ind])
                    psdd.value(ins_inp, clear_data = False)

                    if x_ind % 100 == 0:
                        print("\t\t\tx_ind: ", x_ind)
                    
                    all_samples = []
                    for si in range(sample_size):
                        sample_str = list(str(psdd.simulate_with_evidence(ins_inp)))
                        X_test_cur_sample = np.array(list(map(int, sample_str)))
                        all_samples.append(X_test_cur_sample)
                    
                    sample_features = glc.calculate_features(np.array(all_samples))
                    sample_predict = glc.predict(sample_features)
                    # sample_predict = glc.predict(sample_features)
                    sample_predict_all.append(np.mean(sample_predict))                    
                    psdd.clear_bits()
                
                if not isRegresion:
                    throw("not implemented")
                else:    
                    sample_predict_all = np.array(sample_predict_all)    
                    print(sample_predict_all)        
                    sample_predict_aggregated = sample_predict_all #np.mean(sample_predict_all, axis=0)
                        

                cur_sample.append(function(y_test, sample_predict_aggregated))
                predictions[("sample", ki, R)] = deepcopy(sample_predict_aggregated)

                sample_end_t = perf_counter()
                diff_time = sample_end_t - sample_start_t
                logging.info(
                f'did {sample_size} samples in exp in {diff_time} secs')

            median_features = glc.calculate_features(X_test_median)
            mean_features = glc.calculate_features(X_test_mean)

            median_predict = glc.predict(median_features)
            mean_predict = glc.predict(mean_features)

            cur_lr_median.append(function(y_test, median_predict))
            cur_lr_mean.append(function(y_test, mean_predict))
            # cur_em_impute.append  ( function(y_test, clf.predict(X_test_em_impute)))
            # cur_mice_impute.append( function(y_test, clf.predict(X_test_mice_impute)))
            mom_start_t = perf_counter()
            yHat = predict_batch(psdd, glc, X_test_circuit, T,
                                 prob=False, is_regression=isRegresion)
            mom_end_t = perf_counter()

            diff_time = mom_end_t - mom_start_t
            estimate_time = diff_time * (repeat * (len(K) - ki) + (repeat - R))
            logging.info(
                f'computed moment exp in {diff_time} secs, estimated remaining {estimate_time}')

            print(yHat.shape)

            for zzz in range(T + 1):
                predictions[("circuit_{}".format(zzz), ki, R)] = yHat[zzz]
                temp_z = function(y_test, yHat[zzz])
                print("Error or accuracy [T={}] = {}".format(zzz, temp_z))
                cur_circuit_t[zzz].append(deepcopy(temp_z))

            cur_circuit.append(function(y_test, yHat[-1]))
            print("Setting ki={}, R={}".format(ki, R))
            predictions[("circuit", ki, R)] = deepcopy(yHat[-1])
            predictions[("median", ki, R)] = deepcopy(median_predict)
            predictions[("mean", ki, R)] = deepcopy(mean_predict)

        k_all.append(k)
        missing_err_lr_median_all.append(cur_lr_median)
        missing_err_lr_mean_all.append(cur_lr_mean)
        # missing_err_lr_em_impute_all.append(cur_em_impute)
        # missing_err_lr_mice_impute_all.append(cur_mice_impute)
        missing_err_circuit_all.append(cur_circuit)
        missing_err_sample_all.append(cur_sample)

        for zyz in range(T + 1):
            missing_err_circuit_all_t[zyz].append(deepcopy(cur_circuit_t[zyz]))

        # print(missing_err_circuit_all_t)

    data = {
        "circuit":   missing_err_circuit_all,
        "median": np.array(missing_err_lr_median_all),
        "mean":    np.array(missing_err_lr_mean_all),
        "features_count": FEATURES.shape[0],
        "k":     np.array(k_all),
        "repeat": repeat,
        "missing_all": missing_all,
        "predictions": predictions,
        "y_true": y_test,
        "sample_size": sample_size,
        "sample": np.array(missing_err_sample_all),
        # "em_impute": np.array(missing_err_lr_em_impute_all),
        # "mice_impute": np.array(missing_err_lr_mice_impute_all),
    }

    for it3 in range(T + 1):
        data["circuit_{}".format(it3)] = missing_err_circuit_all_t[it3]

    return data


def plot_results_paper(data, yTrue=None, setting={}):
    import matplotlib.pyplot as plt

    import matplotlib
    matplotlib.rcParams.update({'errorbar.capsize': 3})
    matplotlib.rcParams.update({'figure.autolayout': True})
    matplotlib.rcParams.update({'lines.linewidth': 1.5})
    matplotlib.rcParams.update({'legend.fontsize': 30})

    m_markersize = 7

    matplotlib.rcParams['ps.useafm'] = True
    #matplotlib.rcParams['pdf.use14corefonts'] = True
    # matplotlib.rcParams['text.usetex'] = True

    K = data["k"]
    font = {'size': 32}
    plt.rc('font', **font)

    SIZE = setting["size"] if "size" in setting else (8, 6)
    plt.figure(figsize=SIZE)

    percentage = setting["percentage"] if "percentage" in setting else False
    saveAs = setting["saveAs"] if "saveAs" in setting else "plot.pdf"
    Ylabel = setting["Ylabel"] if "Ylabel" in setting else "Accuracy"
    Xlabel = setting["Xlabel"] if "Xlabel" in setting else "% Missing"
    title = setting["title"] if "title" in setting else "MNIST"
    multiply = setting["mult"] if "mult" in setting else 1.0
    function = setting["function"] if "function" in setting else None
    show = set(setting["show"]) if "show" in setting else set(["mean", "median", "circuit"])

    if (not function is None) and (yTrue is None):
        raise Exception("If function is specified yTrue should also be specified.")

    subset = setting["subset"] if "subset" in setting else np.ones(len(K), dtype='bool')

    legendInclude = setting["legend"] if "legend" in setting else True
    features_count = data["features_count"] if "features_count" in data else 1.0
    plt.title(title)

    choices = [
        "mean",
        "median",
        "circuit",
        "circuit_0",
        "circuit_1"
    ]
    labels = [
        "Mean",
        "Median",
        r"$M_{1}$ (ours)",# Circuit",
        "Circuit T0",
        r"${T}_{1}$ (ours)",# Circuit",
    ]
    fmts = [
        "o--",#"bo--",
        "+-.",#"m+-.",
        "x-",#"rx-",
        "x-",#"rx-",
        "x-",#"rx-",
    ]

    extra_fmts = [
        "v-.",
        "^-.",
        "^-.",
    ]

    color_dict = {
        "circuit" : "#B71C1C",
        "circuit_0" : "#C62828",
        "circuit_1" : "#D32F2F",
        "circuit_2" : "#E53935",
        "circuit_3" : "#F44336",
        "circuit_4" : "#EF9A9A",
        "mpe": "#1D2DE0",
        # "mpe": (30/255.0, 132/255.0, 149/255.0),
        "mice": "#795548",
        "median": "#FFBD2A",
        "mean": "#00695C",
        "sample": "#EF9A9A",
    }

    

    # if not percentage:
    K = np.copy(K[subset]) / (0.01 * features_count)

    KC = dict()

    if not function:
        plot_data = data
    else:
        plot_data = dict()

        for c in show:
            KC[c] = deepcopy([])
            maink_list = []
            for ki, k in enumerate(data["k"]):
                curR_list = []
                for R in range(data["repeat"]):
                    # print(c, ki, R)
                    if (c, ki, R) in data["predictions"]:
                        # print("\t inside")
                        cur_pred = data["predictions"][(c, ki, R)]
                        curR_list.append(function(cur_pred, yTrue))

                if len(curR_list) > 0:
                    maink_list.append(deepcopy(curR_list))
                    KC[c].append(k)

            plot_data[c] = deepcopy(maink_list)

    # for i,c in enumerate(choices):
    #     if c in data and c in show:
    #         plt.errorbar(K, multiply*np.mean(data[c], axis=1), yerr = multiply*np.std(data[c], axis=1), label=labels[i], fmt=fmts[i] )
    # print(show)
    if ("regression" not in setting) or setting["regression"]:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    for i, c in enumerate(show):
        #print(c, plot_data[c])
        if c in choices:
            #print(plot_data[c])
            cur_data = np.array(plot_data[c])[subset]
            idx = choices.index(c)
            MEAN = multiply * np.mean(cur_data, axis=1)
            STD  = multiply * np.std(cur_data, axis=1)

            # print("Results for ", c)
            # print(MEAN)
            # print(STD)

            plt.errorbar(K, MEAN, yerr=STD, label=labels[idx], fmt=fmts[idx], c=color_dict[c], markersize = m_markersize)
        else:
            cur_data = plot_data[c]
            # print(cur_data)
            MEAN = np.array([np.mean(cur_data[i]) for i in range(len(cur_data))], dtype='float')
            STD = np.array([np.std(cur_data[i]) for i in range(len(cur_data))], dtype='float')
            
            # print("Results for ", c)
            # print(MEAN)
            # print(STD)

            cur_kk = np.array(KC[c]) / (0.01 * features_count)
            #print(cur_kk)
            cur_label = c.capitalize()#.replace("_", " T")
            if cur_label == "Mpe":
                cur_label = "MPE"

            plt.errorbar(cur_kk, multiply * MEAN, yerr=multiply * STD,
                         label=cur_label, fmt=extra_fmts[i % len(extra_fmts)], c=color_dict[c], markersize = m_markersize)

    

    if Ylabel != "":
        plt.ylabel(Ylabel)
    if Xlabel != "":
        plt.xlabel(Xlabel)
    if legendInclude:
        plt.legend(loc='best', fontsize='x-small')
    plt.savefig(saveAs)
    return plt


def do_other_impute(X_test, y_test, psdd, glc, setting, missing_exp_data, other_result_path=None):

    X_impute_mean = np.mean(X_test, axis=0)

    missing_err_lr_em_impute_all = []
    missing_err_lr_mice_impute_all = []
    missing_err_lr_psddmpe_impute_all = []
    missing_err_lr_mean_all = []
    missing_err_sample_all = []

    # isRegresion = setting["regression"] if "regression" in setting else False
    percentage = setting["percentage"] if "percentage" in setting else False
    do_psdd_mpe = setting["psddmpe"] if "psddmpe" in setting else False
    do_emImpute = setting["emImpute"] if "emImpute" in setting else False
    do_miceImpute = setting["miceImpute"] if "miceImpute" in setting else False
    do_sample = setting["sampleImpute"] if "sampleImpute" in setting else False
    function = setting["function"] if "function" in setting else None

    # if do_sample:
    if "sampleSize" in setting:
        sample_size = int(setting["sampleSize"])
    else:
        sample_size = 100

    generate_new_missing = setting["generate_new_missing"] if "generate_new_missing" in setting else False

    other_predictions = dict()  # (model, k, R)
    k_all_other = []

    print("Using following function: ", function)
    FEATURES = np.array([i for i in range(X_test.shape[1])])

    if generate_new_missing:
        cur_k = setting["k_other"]
        repeat = setting["repeat"]
    else:
        cur_k = missing_exp_data["k"]
        repeat = missing_exp_data["repeat"]

    print("K: ", cur_k)
    print("Number of feautres: {}".format(FEATURES.shape[0]))

    for ki, k_percent in enumerate(cur_k):
        # if percentage:
        #     k = int((k_percent / 100.0) * FEATURES.shape[0])
        # else:
        k = k_percent

        skip_this = (k == FEATURES.shape[0])

        if k > FEATURES.shape[0]:
            print("Early stop: Only had {} features possible to remove vs {}".format(FEATURES.shape[0], k))
            break

        cur_lr_mean = []
        cur_psddmpe_impute = []
        cur_em_impute = []
        cur_mice_impute = []
        cur_sample = []

        for R in range(repeat):
            print("K = {}, R = {}".format(k, R))

            X_test_mean = np.array(X_test, dtype='float')
            X_test_psdd_impute = np.array(X_test, dtype='float')
            X_test_em_impute = np.array(X_test, dtype='float')
            X_test_mice_impute = np.array(X_test, dtype='float')
            #X_test_sample = np.array(X_test, dtype='float')
            X_test_sample = np.array(X_test, dtype='int')

            if not generate_new_missing:
                cur_miss_mask = missing_exp_data["missing_all"][(ki, R)]
            else:
                cur_miss_mask = np.zeros(X_test.shape, dtype=bool)
                for i in range(X_test.shape[0]):
                    miss = np.random.choice(FEATURES, k, replace=False)
                    cur_miss_mask[i][miss] = True

            print(X_impute_mean.shape, X_test_mean.shape)
            for ii in range(X_test_mean.shape[0]):
                X_test_mean[ii][cur_miss_mask[ii]] = X_impute_mean[cur_miss_mask[ii]]

            
            X_test_psdd_impute[cur_miss_mask] = -1
            X_test_em_impute[cur_miss_mask] = np.nan
            X_test_mice_impute[cur_miss_mask] = np.nan
            X_test_sample[cur_miss_mask] = -1

            print("Missing counts: ", np.mean(np.sum(cur_miss_mask, axis=1)))

            if do_psdd_mpe:
                print("Strating to do mpe method")
                start = perf_counter()
                # psdd.mpe
                for i in range(X_test_psdd_impute.shape[0]):
                    Nf = X_test_psdd_impute.shape[1]

                    evidence = Inst.from_list(X_test_psdd_impute[i], Nf, zero_indexed=True)
                    val, TEMP = psdd.mpe(evidence)

                    TEMP = [int(chr) for chr in str(TEMP).split(" ")[0]]
                    # print(TEMP)
                    X_test_psdd_impute[i, :] = np.array(TEMP)

                end = perf_counter()
                estimate_time = (end - start) * (repeat * (len(cur_k) - ki) + (repeat - R))
                print("\tDone imputing! " + str(end - start))
                print(f'estimated remaining {estimate_time}')
            else:
                X_test_psdd_impute = np.zeros(X_test.shape)

            if do_emImpute and k > 0 and not skip_this:
                import impyute
                start = perf_counter()
                loops = 6
                print("\tStarting to em impute with loops = {}".format(loops))
                X_test_em_impute = impyute.em(X_test_em_impute, loops=loops)
                X_test_em_impute[X_test_em_impute < 0.0] = 0.0
                end = perf_counter()
                print("\tDone imputing! " + str(end - start))
            elif k != 0:
                X_test_em_impute = np.zeros(X_test.shape)

            MICE_FAILED = False
            if do_miceImpute and k > 0 and not skip_this:
                start = perf_counter()
                MICE_FAILED = False
                print("\tStarting to mice impute {}".format(X_test_mice_impute.shape))

                # X_test_mice_impute = impyute.mice(X_test_mice_impute)
                try:
                    df = pd.DataFrame(X_test_mice_impute, columns=[
                                      "a" + str(i) for i in range(X_test_mice_impute.shape[1])])
                    imp = mice.MICEData(df)
                    imp.update_all(1)

                    # print(imp.next_sample().values)
                    # print(imp.next_sample().values.shape)

                    X_test_mice_impute = np.copy(imp.next_sample().values)
                except Exception as inst:
                    print("Failed mice on {}, {}".format(ki, R))
                    # raise inst
                    print(type(inst))
                    print(inst)
                    MICE_FAILED = True

                X_test_mice_impute[X_test_mice_impute < 0.0] = 0.0
                X_test_mice_impute[X_test_mice_impute > 1.0] = 1.0

                end = perf_counter()

                estimate_time = (end - start) * (repeat * (len(cur_k) - ki) + (repeat - R))
                print("\tDone imputing! " + str(end - start))
                print(f'estimated remaining {estimate_time}')

            elif k != 0:
                X_test_mice_impute = np.zeros(X_test.shape)

            if do_sample:   
                print("Doing sampling with {} samples".format(sample_size))
                sample_start_t = perf_counter()
            
                sample_predict_all = [] 
                for x_ind in range(len(y_test)):
                    ins_inp = InstMap.from_list(X_test_sample[x_ind])
                    psdd.value(ins_inp, clear_data = False)

                    if x_ind % 100 == 0:
                        print("\t\t\tx_ind: ", x_ind)
                    
                    all_samples = []
                    for si in range(sample_size):
                        sample_str = list(str(psdd.simulate_with_evidence(ins_inp)))
                        X_test_cur_sample = np.array(list(map(int, sample_str)))
                        all_samples.append(X_test_cur_sample)
                    
                    sample_features = glc.calculate_features(np.array(all_samples))
                    sample_predict = glc.predict(sample_features)
                    # sample_predict = glc.predict(sample_features)
                    sample_predict_all.append(np.mean(sample_predict))                    
                    psdd.clear_bits()
                

                sample_predict_all = np.array(sample_predict_all)    
                # print(sample_predict_all)        
                sample_predict_aggregated = sample_predict_all #np.mean(sample_predict_all, axis=0)
                        

                cur_sample.append(function(y_test, sample_predict_aggregated))
                other_predictions[("sample", ki, R)] = deepcopy(sample_predict_aggregated)

                sample_end_t = perf_counter()
                diff_time = sample_end_t - sample_start_t
                logging.info(
                f'did {sample_size} samples in exp in {diff_time} secs')



            mean_features = glc.calculate_features(X_test_mean)
            mean_predict = glc.predict(mean_features)
            mean_f = function(y_test, mean_predict)
            cur_lr_mean.append(mean_f)
            other_predictions[("mean", ki, R)] = deepcopy(mean_predict)

            psddmpe_features = glc.calculate_features(X_test_psdd_impute)
            if not MICE_FAILED:
                mice_features = glc.calculate_features(X_test_mice_impute)
            em_features = glc.calculate_features(X_test_em_impute)

            psddmpe_predict = glc.predict(psddmpe_features)
            if not MICE_FAILED:
                mice_predict = glc.predict(mice_features)
            em_predict = glc.predict(em_features)

            other_predictions[("mpe", ki, R)] = deepcopy(psddmpe_predict)
            if not MICE_FAILED:
                other_predictions[("mice", ki, R)] = deepcopy(mice_predict)
            other_predictions[("em", ki, R)] = deepcopy(em_predict)

            mpe_f = function(y_test, psddmpe_predict)
            if not MICE_FAILED:
                mice_f = function(y_test, mice_predict)

            em_f = function(y_test, em_predict)

            print("\tMPE_f error {}".format(mpe_f))
            if not MICE_FAILED:
                print("\tMICE error {}".format(mice_f))
            print("\tEM   error {}".format(em_f))
            print("\tMean error {}".format(mean_f))  
            
            cur_psddmpe_impute.append(mpe_f)
            if not MICE_FAILED:
                cur_mice_impute.append(mice_f)
            cur_em_impute.append(em_f)

            if other_result_path is not None:
                print("Saving this batch to file {},{} --> {}".format(ki, R, other_result_path))
                last_batch_other_data = {
                    "cur_batch": ki,
                    "did_em": do_emImpute,
                    "did_mice": do_miceImpute,
                    "other_k": k_all_other,
                    "other_predictions": other_predictions,
                    "em_impute": np.array(missing_err_lr_em_impute_all),
                    "mice_impute": np.array(missing_err_lr_mice_impute_all),
                    "mpe_impute": np.array(missing_err_lr_psddmpe_impute_all),
                    "mean_impute": missing_err_lr_mean_all,
                    "sample_impute": np.array(missing_err_sample_all),
                }
                with open(other_result_path, "wb") as outfile:
                    pickle.dump(last_batch_other_data, outfile)

        k_all_other.append(k)
        missing_err_lr_mean_all.append(cur_lr_mean)
        missing_err_lr_em_impute_all.append(cur_em_impute)
        missing_err_lr_mice_impute_all.append(cur_mice_impute)
        missing_err_lr_psddmpe_impute_all.append(cur_psddmpe_impute)
        missing_err_sample_all.append(cur_sample)

    other_data = {
        "did_em": do_emImpute,
        "did_mice": do_miceImpute,
        "did_sample": do_sample,
        "other_k": k_all_other,
        "other_predictions": other_predictions,
        "em_impute": np.array(missing_err_lr_em_impute_all),
        "mice_impute": np.array(missing_err_lr_mice_impute_all),
        "mpe_impute": np.array(missing_err_lr_psddmpe_impute_all),
        "mean_impute": missing_err_lr_mean_all,
        "sample_size": sample_size,
        "sample_impute": np.array(missing_err_sample_all),
    }

    return other_data
