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
from numpy.testing import assert_array_almost_equal

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import Ridge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from LogisticCircuit.structure.Vtree import generate_random_vtree
from LogisticCircuit.algo.LogisticCircuit import learn_logistic_circuit
from LogisticCircuit.algo.LogisticRegression import LogisticRegression
from LogisticCircuit.util.DataSet import DataSet
from LogisticCircuit.algo.RegressionCircuit import learn_regression_circuit, RegressionCircuit

LEARN_PSDD_CMD = 'export LD_LIBRARY_PATH="{}";java -jar {} learnPsdd search -v {} -m l-1 -o {} -d {} -b {} -e {}'


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

    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Regularization coefficient")

    parser.add_argument("--n-iter-sl", type=int, default=5000)
    parser.add_argument("--n-iter-pl", type=int, default=15)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--psdd-n-iter", type=int, default=100)

    parser.add_argument("--vtree", type=str, default="balanced",
                        help="Path for vtree or mode to get it")

    parser.add_argument('--psdd-jar', type=str,
                        default='./exp/psdd/psdd.jar',
                        help='psdd jar path')

    parser.add_argument('--psdd-jar-path', type=str,
                        default='./exp/psdd/',
                        help='psdd jar dir path')

    parser.add_argument('--regression', action='store_true',
                        help='Regression instead of classification')

    parser.add_argument('--baseline', action='store_true',
                        help='Regression instead of classification')

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
    # learn logistic circuit
    #
    # TODO: accomodate also for a regression circuit
    if not args.regression:
        n_classes = np.max(y_train) + 1

    one_hot = not args.regression
    train_data = DataSet(x_train, y_train, one_hot)
    valid_data = DataSet(x_valid, y_valid, one_hot)
    test_data = DataSet(x_test, y_test, one_hot)

    if args.regression:
        if args.baseline:
            logging.info('Training vanilla Ridge regression as a baseline')
            model = Ridge(alpha=args.alpha,
                          fit_intercept=False,
                          normalize=False,
                          copy_X=True,
                          max_iter=args.n_iter_pl,
                          tol=1e-5,
                          solver='auto',
                          # coef_=self._parameters,
                          random_state=rand_gen,
                          )
            model.fit(x_train, y_train)

            y_pred = model.predict(x_train)
            train_mse = mean_squared_error(y_train, y_pred)
            logging.info(f'\t\ttrain mse: {train_mse:.5f}')

            y_pred = model.predict(x_valid)
            valid_mse = mean_squared_error(y_valid, y_pred)
            logging.info(f'\t\tvalid mse: {valid_mse:.5f}')

            y_pred = model.predict(x_test)
            test_mse = mean_squared_error(y_test, y_pred)
            logging.info(f'\t\ttest mse: {test_mse:.5f}')

            #
            # and save
            baseline_path = os.path.join(out_path, 'ridge.pklz')
            with gzip.open(baseline_path, 'wb') as f:
                pickle.dump(model, f)

        circuit, train_history = learn_regression_circuit(vtree=v,
                                                          train=train_data,
                                                          max_iter_sl=args.n_iter_sl,
                                                          max_iter_pl=args.n_iter_pl,
                                                          depth=args.depth,
                                                          alpha=args.alpha,
                                                          num_splits=args.n_splits,
                                                          rand_gen=rand_gen,
                                                          )
    else:
        if args.baseline:
            logging.info('Training vanilla logistic regression as a baseline')
            model = LogisticRegression(solver="saga",
                                       fit_intercept=False,
                                       multi_class="ovr",
                                       max_iter=args.n_iter_pl,
                                       C=args.alpha,
                                       tol=1e-5,
                                       random_state=rand_gen)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_train)
            train_acc = accuracy_score(y_train, y_pred)
            logging.info(f'\t\ttrain accuracy: {train_acc:.5f}')

            y_pred = model.predict(x_valid)
            valid_acc = accuracy_score(y_valid, y_pred)
            logging.info(f'\t\tvalid accuracy: {valid_acc:.5f}')

            y_pred = model.predict(x_test)
            test_acc = accuracy_score(y_test, y_pred)
            logging.info(f'\t\ttest accuracy: {test_acc:.5f}')

            #
            # and save
            baseline_path = os.path.join(out_path, 'lr.pklz')
            with gzip.open(baseline_path, 'wb') as f:
                pickle.dump(model, f)

        circuit, train_history = learn_logistic_circuit(vtree=v,
                                                        n_classes=n_classes,
                                                        train=train_data,
                                                        C=args.alpha,
                                                        max_iter_sl=args.n_iter_sl,
                                                        max_iter_pl=args.n_iter_pl,
                                                        depth=args.depth,
                                                        num_splits=args.n_splits,
                                                        rand_gen=rand_gen,
                                                        )

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

    # pre_save_params = np.copy(circuit.parameters)
    # pre_bias = np.copy(circuit._bias)
    # logging.info(f'PRE saved params {pre_save_params}')

    #
    # save circuit
    circuit_path = os.path.join(out_path, f'{dataset_name}.glc')
    with open(circuit_path, 'w') as f:
        circuit.save(f)
    logging.info(f'Circuit saved to {circuit_path}')

    # post_save_params = np.copy(circuit.parameters)
    # post_bias = np.copy(circuit._bias)
    # logging.info(f'POST saved params {post_save_params}')
    # assert_array_almost_equal(post_save_params, pre_save_params)

    # #
    # # load it back
    # with open(circuit_path, 'r') as f:
    #     c = RegressionCircuit(v, circuit_file=f)
    # c_post = c.parameters
    # logging.info(f'POST saved params {c_post}')

    # circuit_path = os.path.join(out_path, f'{dataset_name}-resaved.glc')
    # with open(circuit_path, 'w') as f:
    #     c.save(f)
    # logging.info(f'Circuit saved to {circuit_path}')

    # assert_array_almost_equal(c_post, pre_save_params)

    # 0 / 0

    #
    # save training performances
    perf_path = os.path.join(out_path, f'{dataset_name}.train-hist')
    np.save(perf_path, train_history)
    logging.info(f'Training history saved to {train_history}')
    #
    # and plot it
    perf_path = os.path.join(out_path, f'{dataset_name}.train-hist.pdf')
    plt.plot(np.arange(len(train_history)), train_history)
    plt.savefig(perf_path)
    plt.close()

    #
    # create PSDD
    #

    #
    # dump data
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
                               args.psdd_n_iter)

    psdd_path = os.path.join(out_path, f'{dataset_name}.psdd')
    logging.info(f'Executing LearnPSDD\n{psdd_cmd}')
    psdd_start_t = perf_counter()
    os.system(psdd_cmd)
    psdd_end_t = perf_counter()
    logging.info(f'Learned psdd in {psdd_end_t - psdd_start_t} secs')

    CP_PSDD_CMD = f'cp {out_path}/models/final.psdd {psdd_path}'
    os.system(CP_PSDD_CMD)
