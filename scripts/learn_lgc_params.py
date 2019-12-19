import os
import gzip
import numpy as np
import pickle
import logging
import sys
import datetime
from time import perf_counter
import itertools
import argparse
import json
import copy
sys.path.append("LogisticCircuit")
sys.path.append("pypsdd")
sys.path.append(".")

from structure.Vtree import Vtree as LC_Vtree
from algo.LogisticCircuit import LogisticCircuit
from LogisticCircuit.util.DataSet import DataSet

DATASET = 'mnist'
DATASET_PATH = f'data/{DATASET}/{DATASET}.pklz'
# DATASET = 'data/fmnist/fmnist.pklz'

MODEL = f'data/{DATASET}/mnist.circuit'
VTREE = f'data/{DATASET}/{DATASET}.vtree'

OUTPUT = f'exp/class-circuit-grid/{DATASET}/best'

ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
DEPTHS = [2, 10, 20, 100]
SPLITS = [1, 2, 5, 10]
SL_ITERS = [10, 100]
PL_ITERS = [100, 200, 500]

if __name__ == '__main__':

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

    args = parser.parse_args()

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

    v_tree_path = args.dataset.replace('.pklz', '.vtree')
    lc_vtree = LC_Vtree.read(v_tree_path)
    print('\t\t...loaded vtree')

    glc_path = args.dataset.replace('.pklz', '.circuit')
    with open(glc_path) as f:
        orig_circuit = LogisticCircuit(lc_vtree, num_classes=10, circuit_file=f)
    print('\t\t...loaded glc')

    n_classes = np.max(y_train) + 1
    train_data = DataSet(x_train, y_train, one_hot=True)
    valid_data = DataSet(x_valid, y_valid, one_hot=True)
    test_data = DataSet(x_test, y_test, one_hot=True)

    res_logger.info(
        f'alpha\tdepth\tn-splits\tn-iter-sl\tn-iter-pl\ttrain-mse\tvalid-mse\ttest-mse\ttrain-time')
    for config in itertools.product(args.n_iter_pl,
                                    args.alpha):
        logging.info(f'\t considering config:\n\t\t{config}')

        #
        # resetting the seed
        rand_gen = np.random.RandomState(args.seed)

        circuit = copy.deepcopy(orig_circuit)

        n_iter_pl, alpha = config
        config_str = f'A{alpha}-PL{n_iter_pl}'
        config_out_path = os.path.join(out_path, config_str)
        os.makedirs(config_out_path, exist_ok=True)

        # logging.info(f'Structure learning done in {sl_end_t - sl_start_t} secs')

        #
        # lern
        train_data.features = circuit.calculate_features(train_data.images)
        pl_start_t = perf_counter()
        circuit.learn_parameters(train_data, n_iter_pl, C=alpha, rand_gen=rand_gen)
        pl_end_t = perf_counter()

        #
        # evaluate
        # FIXME: change name from images
        train_data.features = circuit.calculate_features(train_data.images)
        train_acc = circuit.calculate_accuracy(train_data)
        logging.info(f'\t\ttrain accuracy: {train_acc:.9f}')

        valid_data.features = circuit.calculate_features(valid_data.images)
        valid_acc = circuit.calculate_accuracy(valid_data)
        logging.info(f'\t\tvalid accuracy: {valid_acc:.9f}')

        test_data.features = circuit.calculate_features(test_data.images)
        test_acc = circuit.calculate_accuracy(test_data)
        logging.info(f'\t\ttest accuracy: {test_acc:.9f}')

        res_logger.info(f'{alpha}\t{n_iter_pl}\t' +
                        f'{train_acc:.9f}\t{valid_acc:.9f}\t{test_acc:.9f}')

        #
        # save circuit
        circuit_path = os.path.join(config_out_path, f'{dataset_name}.glc')
        with open(circuit_path, 'w') as f:
            circuit.save(f)
        logging.info(f'Circuit saved to {circuit_path}')

        #
