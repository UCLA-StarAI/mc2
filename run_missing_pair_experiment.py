import sys
sys.path.append("LogisticCircuit")
sys.path.append("pypsdd")
sys.path.append('..')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

import argparse
import pickle
import gzip

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
import os
import logging
import pickle
import gzip
import json

import numpy as np

from utils_missing import run_missing_exp, plot_results_paper

from LogisticCircuit.structure.Vtree import generate_random_vtree
from LogisticCircuit.algo.LogisticCircuit import learn_logistic_circuit
from LogisticCircuit.util.DataSet import DataSet
from LogisticCircuit.algo.RegressionCircuit import learn_regression_circuit, RegressionCircuit

import psdd_io
from manager import PSddManager
from algo.LogisticCircuit import LogisticCircuit
from structure.Vtree import Vtree as LC_Vtree
from vtree import Vtree as PSDD_Vtree

from sklearn.metrics import mean_squared_error


def dump_data_csv(X, data_path):
    with open(data_path, 'w') as f:
        for x in X:
            f.write('{}\n'.format(','.join(str(s) for s in x)))


if __name__ == '__main__':

    start_all = perf_counter()

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Path to data dir')

    parser.add_argument('-o', '--output', type=str,
                        default='./exp/missing/',
                        help='Output path to exp result')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

    parser.add_argument('--repeat', type=int,
                        default=10,
                        help='How many times to repeat for each missingness')

    parser.add_argument('--taylor', type=int,
                        default=3,
                        help='How many taylor terms (only for classification)')

    parser.add_argument('--miss-perc', type=int, nargs='+',
                        default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='How many taylor terms (only for classification)')

    parser.add_argument("--vtree", type=str, default="balanced",
                        help="Path for vtree or mode to get it")

    # parser.add_argument('--psdd', type=str,
    #                     default=None,
    #                     help='Path to psdd')

    # parser.add_argument('--glc', type=str,
    #                     default=None,
    #                     help='Path to glc (logistic circuit or regression circuit)')

    parser.add_argument('--regression', action='store_true',
                        help='Regression instead of classification')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    #
    # creating output dirs if they do not exist
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = os.path.basename(args.dataset).replace('.pklz', '')

    # if args.exp_id:
    #     out_path = os.path.join(args.output, args.exp_id)
    # else:
    #     out_path = os.path.join(args.output,  '{}_{}'.format(dataset_name, date_string))
    # os.makedirs(out_path, exist_ok=True)
    out_path = args.output

    args_out_path = os.path.join(out_path, 'miss.args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

    out_log_path = os.path.join(out_path,  'miss.exp.log')
    logging.info('Opening log file... {}'.format(out_log_path))

    #
    # loading up datasets
    with gzip.open(args.dataset, 'rb') as f:
        data_splits = pickle.load(f)

    #
    # unpacking splits
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_splits

    if not args.regression:
        y_train = y_train.astype(np.int8)
        y_valid = y_valid.astype(np.int8)
        y_test = y_test.astype(np.int8)

    n_features = x_train.shape[1]
    assert x_valid.shape[1] == n_features
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[1] == n_features
    assert x_valid.shape[0] == y_valid.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    logging.info(f'\nLoaded dataset splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    # load the vtrees
    if args.vtree is None:
        vtree_path = os.path.join(out_path, f'{dataset_name}.vtree')
    else:
        vtree_path = args.vtree

    glc_vtree = LC_Vtree.read(vtree_path)

    #
    # Load logistic/Regression circuit
    # TODO: accomodate also for a regression circuit
    if not args.regression:
        n_classes = np.max(y_train) + 1
    else:
        n_classes = 1

    one_hot = not args.regression
    train_data = DataSet(x_train, y_train, one_hot)
    valid_data = DataSet(x_valid, y_valid, one_hot)
    test_data = DataSet(x_test, y_test, one_hot)

    circuit_path = os.path.join(out_path, f'{dataset_name}.glc')
    if args.regression:
        with open(circuit_path) as circuit_file:
            circuit = RegressionCircuit(glc_vtree, circuit_file=circuit_file)
    else:
        with open(circuit_path) as circuit_file:
            circuit = LogisticCircuit(glc_vtree, n_classes, circuit_file=circuit_file)

    if args.regression:
        #
        # evaluate
        # FIXME: change name from images
        train_data.features = circuit.calculate_features(train_data.images)
        train_acc = circuit.calculate_error(train_data)
        logging.info(f'\t\ttrain error: {train_acc:.5f}')

        valid_data.features = circuit.calculate_features(valid_data.images)
        valid_acc = circuit.calculate_error(valid_data)
        logging.info(f'\t\tvalid error: {valid_acc:.5f}')

        test_data.features = circuit.calculate_features(test_data.images)
        test_acc = circuit.calculate_error(test_data)
        logging.info(f'\t\ttest error: {test_acc:.5f}')
    else:
        #
        # evaluate
        # FIXME: change name from images
        train_data.features = circuit.calculate_features(train_data.images)
        train_acc = circuit.calculate_accuracy(train_data)
        logging.info(f'\t\ttrain accuracy: {train_acc:.5f}')

        valid_data.features = circuit.calculate_features(valid_data.images)
        valid_acc = circuit.calculate_accuracy(valid_data)
        logging.info(f'\t\tvalid accuracy: {valid_acc:.5f}')

        test_data.features = circuit.calculate_features(test_data.images)
        test_acc = circuit.calculate_accuracy(test_data)
        logging.info(f'\t\ttest accuracy: {test_acc:.5f}')

    # load PSDD
    vtree_psdd = PSDD_Vtree.read(vtree_path)
    manager = PSddManager(vtree_psdd)
    psdd_path = os.path.join(out_path, f'{dataset_name}.psdd')
    psdd = psdd_io.psdd_yitao_read(psdd_path, manager)

    def sqrt_mse(x, y): return np.sqrt(mean_squared_error(x, y))

    REPEAT = int(args.repeat) if args.repeat is not None else 10
    TAYLOR = int(args.taylor) if args.taylor is not None else 3

    setting = {
        'k': args.miss_perc,
        'percentage': True,
        'repeat': REPEAT,
        'T': TAYLOR,
        'sample_method': False,  # whether or not run monte carlo estimation
        'sample_size': 200,
    }

    if args.regression:
        setting["function"] = sqrt_mse
        setting["regression"] = args.regression
        setting['T'] = 0

    print("----SETTING----")
    print(setting)
    print("---------------")

    result = run_missing_exp(x_test, y_test, psdd, circuit, setting)
    result_path = os.path.join(out_path, f'{dataset_name}_{args.exp_id}_missing_result.pickle')

    result_path_z = result_path + "z"
    logging.info("Writing results into {}".format(result_path_z))
    with gzip.open(result_path_z, "wb") as gfile:
        pickle.dump(result, gfile)

    end_all = perf_counter()
    logging.info("Total time: {}".format(end_all - start_all))
