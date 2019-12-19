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

import numpy as np

from LogisticCircuit.structure.Vtree import generate_random_vtree

LEARN_PSDD_CMD = 'export LD_LIBRARY_PATH="{}";java -jar {} learnPsdd search -v {} -m l-1 -o {} -d {} -b {} -e {} -p {}'


def dump_data_csv(X, data_path):
    with open(data_path, 'w') as f:
        for x in X:
            f.write('{}\n'.format(','.join(str(s) for s in x)))


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

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

    parser.add_argument("--psdd-n-iter", type=int, default=100)

    parser.add_argument("--vtree", type=str, default="balanced",
                        help="Path for vtree or mode to get it")

    parser.add_argument('--psdd-jar', type=str,
                        default='./exp/psdd/psdd.jar',
                        help='psdd jar path')

    parser.add_argument('--psdd-path', type=str,
                        default='',
                        help='psdd path as init')

    parser.add_argument('--psdd-jar-path', type=str,
                        default='./exp/psdd/',
                        help='psdd jar dir path')

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
    file_handler = logging.FileHandler("{0}/{1}.learnpsdd.log".format(log_dir, 'exp'))
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

    args_out_path = os.path.join(out_path, 'learnpsdd.args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

    out_log_path = os.path.join(out_path,  'learnpsdd.exp.log')
    logging.info('Opening log file... {}'.format(out_log_path))

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
        vtree_path = os.path.join(out_path, f'{dataset_name}.vtree')
        v = generate_random_vtree(n_vars=n_features, rand_gen=rand_gen, balanced=True)
        v.save(vtree_path)
        logging.info(f'Generated random balanced vtree and saved to {vtree_path}')
    else:
        vtree_path = args.vtree
        logging.info(f'Loaded vtree from {vtree_path}')

    #
    # wrapping learnPSDD
    #

    #
    # dump data in learnPSDD format
    x_train_int = x_train.astype(np.int8)
    train_data_path = os.path.join(out_path, f'{dataset_name}.train')
    dump_data_csv(x_train_int, train_data_path)

    x_valid_int = x_valid.astype(np.int8)
    valid_data_path = os.path.join(out_path, f'{dataset_name}.valid')
    dump_data_csv(x_valid_int, valid_data_path)

    x_test_int = x_test.astype(np.int8)
    test_data_path = os.path.join(out_path, f'{dataset_name}.test')
    dump_data_csv(x_test_int, test_data_path)

    logging.info('Dumped data in learnPSDD format')

    psdd_cmd = str(LEARN_PSDD_CMD)
    psdd_cmd = psdd_cmd.format(args.psdd_jar_path,
                               args.psdd_jar,
                               vtree_path,
                               out_path,
                               train_data_path,
                               valid_data_path,
                               args.psdd_n_iter,
                               args.psdd_path)

    psdd_path = os.path.join(out_path, f'{dataset_name}.psdd')
    logging.info(f'Executing LearnPSDD\n{psdd_cmd}')
    psdd_start_t = perf_counter()
    os.system(psdd_cmd)
    psdd_end_t = perf_counter()
    logging.info(f'Learned psdd in {psdd_end_t - psdd_start_t} secs')

    CP_PSDD_CMD = f'cp {out_path}/models/final.psdd {psdd_path}'
    os.system(CP_PSDD_CMD)
