import sys
sys.path.append("LogisticCircuit")
sys.path.append("pypsdd")
sys.path.append('.')

import argparse

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
import itertools

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from LogisticCircuit.structure.Vtree import generate_random_vtree
from LogisticCircuit.algo.LogisticCircuit import learn_logistic_circuit
from LogisticCircuit.util.DataSet import DataSet
from LogisticCircuit.algo.RegressionCircuit import learn_regression_circuit


def dump_data_csv(X, data_path):
    with open(data_path, 'w') as f:
        for x in X:
            f.write('{}\n'.format(','.join(str(s) for s in x)))


ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
DEPTHS = [2, 10, 20, 100]
SPLITS = [1, 2, 5, 10]
SL_ITERS = [10, 100]
PL_ITERS = [100, 200]

if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Path to data dir')

    parser.add_argument('-o', '--output', type=str,
                        default='./exp/',
                        help='Output path to exp result')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--validate-every', type=int, nargs='?',
                        default=5,
                        help='Validate for early stopping')

    parser.add_argument('--patience', type=int, nargs='?',
                        default=2,
                        help='Patience for early stopping')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

    parser.add_argument("--alpha", type=float, nargs='+',  default=ALPHAS,
                        help="Regularization coefficient")

    parser.add_argument("--n-iter-sl", type=int, nargs='+', default=SL_ITERS)
    parser.add_argument("--n-iter-pl", type=int, nargs='+', default=PL_ITERS)
    parser.add_argument("--depth", type=int, nargs='+', default=DEPTHS)
    parser.add_argument("--n-splits", type=int, nargs='+', default=SPLITS)

    parser.add_argument("--vtree", type=str, default="balanced",
                        help="Path for vtree or mode to get it")

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    #
    # creating output dirs if they do not exist
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = os.path.basename(args.dataset).replace('.pklz', '')

    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
    else:
        out_path = os.path.join(args.output,  '{}_{}'.format(dataset_name, date_string))
    os.makedirs(out_path, exist_ok=True)

    #
    # Logging
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s")
    root_logger = logging.getLogger()

    # to file
    log_dir = os.path.join(out_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler("{0}/{1}.log".format(log_dir, 'exp'))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # and to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    #
    # setting verbosity level
    if args.verbose == 1:
        root_logger.setLevel(logging.INFO)
    elif args.verbose == 2:
        root_logger.setLevel(logging.DEBUG)

    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

    out_log_path = os.path.join(out_path,  'exp.log')
    logging.info('Opening log file... {}'.format(out_log_path))

    res_logger = logging.getLogger('res')
    res_path = os.path.join(out_path,  'exp.res')
    res_file_handler = logging.FileHandler(res_path)
    res_logger.addHandler(res_file_handler)

    logging.info('Opening res file... {}'.format(res_path))

    #
    # loading up datasets
    with gzip.open(args.dataset, 'rb') as f:
        data_splits = pickle.load(f)

    #
    # unpacking splits
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_splits

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

    #
    # load the vtree or create it from scratch
    if args.vtree == 'balanced':
        v = generate_random_vtree(n_vars=n_features, rand_gen=rand_gen, balanced=True)
    vtree_path = os.path.join(out_path, f'{dataset_name}.vtree')
    v.save(vtree_path)

    train_data = DataSet(x_train, y_train, one_hot=False)
    valid_data = DataSet(x_valid, y_valid, one_hot=False)
    test_data = DataSet(x_test, y_test, one_hot=False)

    res_logger.info(
        f'alpha\tdepth\tn-splits\tn-iter-sl\tn-iter-pl\ttrain-mse\tvalid-mse\ttest-mse\ttrain-time')
    for config in itertools.product(args.n_iter_sl,
                                    args.n_iter_pl,
                                    args.depth,
                                    args.alpha,
                                    args.n_splits):
        logging.info(f'\t considering config:\n\t\t{config}')

        #
        # resetting the seed
        rand_gen = np.random.RandomState(args.seed)

        n_iter_sl, n_iter_pl, depth, alpha, n_splits = config
        config_str = f'A{alpha}-D{depth}-S{n_splits}-SL{n_iter_sl}-PL{n_iter_pl}'
        config_out_path = os.path.join(out_path, config_str)
        os.makedirs(config_out_path, exist_ok=True)

        sl_start_t = perf_counter()
        circuit, train_history = learn_regression_circuit(vtree=v,
                                                          train=train_data,
                                                          valid=valid_data,
                                                          max_iter_sl=n_iter_sl,
                                                          max_iter_pl=n_iter_pl,
                                                          depth=depth,
                                                          alpha=alpha,
                                                          validate_every=args.validate_every,
                                                          patience=args.patience,
                                                          num_splits=n_splits,
                                                          rand_gen=rand_gen
                                                          )
        sl_end_t = perf_counter()
        logging.info(f'Structure learning done in {sl_end_t - sl_start_t} secs')

        #
        # evaluate
        # FIXME: change name from images
        train_data.features = circuit.calculate_features(train_data.images)
        train_acc = circuit.calculate_error(train_data)
        logging.info(f'\t\ttrain error: {train_acc:.9f}')

        valid_data.features = circuit.calculate_features(valid_data.images)
        valid_acc = circuit.calculate_error(valid_data)
        logging.info(f'\t\tvalid error: {valid_acc:.9f}')

        test_data.features = circuit.calculate_features(test_data.images)
        test_acc = circuit.calculate_error(test_data)
        logging.info(f'\t\ttest error: {test_acc:.9f}')

        res_logger.info(f'{alpha}\t{depth}\t{n_splits}\t{n_iter_sl}\t{n_iter_pl}\t' +
                        f'{train_acc:.9f}\t{valid_acc:.9f}\t{test_acc:.9f}\t{sl_end_t - sl_start_t}')

        #
        # save circuit
        circuit_path = os.path.join(config_out_path, f'{dataset_name}.glc')
        with open(circuit_path, 'w') as f:
            circuit.save(f)
        logging.info(f'Circuit saved to {circuit_path}')

        #
        # save training performances
        perf_path = os.path.join(config_out_path, f'{dataset_name}.train-hist')
        np.save(perf_path, train_history)
        logging.info(f'Training history saved to {train_history}')
        #
        # and plot it
        perf_path = os.path.join(config_out_path, f'{dataset_name}.train-hist.pdf')
        plt.plot(np.arange(len(train_history) - 1), train_history[1:])
        plt.savefig(perf_path)
        plt.close()
