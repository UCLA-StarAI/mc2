import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
import os
import sys
import logging
import pickle
import gzip
import json
import itertools

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from pprint import pprint

HYPERPARAMS = {
    'ridge': {
        'alpha': [0.001, 0.01, 0.1, 1, 2, 10],
        'fit_intercept': [False],
        'normalize': [False],
        'copy_X': [True],
        'max_iter': [1000],
        'tol': [1e-5],
        'solver': ['saga'],
    },

    'dtr': {
        'criterion': ['mse'],
        'splitter': ['best'],
        'max_depth': [None, 10, 20, 50, 100],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'max_features': [None],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0.0],
        'min_impurity_split': [None],
    },

    'rf': {
        'n_estimators': [10, 20, 100, 200, 1000],
        'criterion': ['mse'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'max_features': ['auto'],
        'max_leaf_nodes': [None],
        # 'min_impurity_decrease': [0.0],
        'min_impurity_split': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'n_jobs': [2],
    },

    'mlp': {
        'hidden_layer_sizes': [(128, ),
                               (128, 128),
                               (256,), (256, 256), (256, 128, 256),
                               (512), (512, 512), (512, 256, 512)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'batch_size': ['auto'],
        'learning_rate': ['constant'],
        'learning_rate_init': [0.001],
        'power_t': [0.5],
        'max_iter': [1000],
        'shuffle': [True],
        'tol': [0.0001],
        'momentum': [0.9],
        'nesterovs_momentum': [True],
        'early_stopping': [False],
        'validation_fraction': [0.1],
        'beta_1': [0.9], 'beta_2': [0.999],
        'epsilon': [1e-08], 'n_iter_no_change': [10]
    }
}

MODELS = {
    'ridge': Ridge,
    'dtr': DecisionTreeRegressor,
    'rf': RandomForestRegressor,
    'mlp': MLPRegressor,

}


def cartesian_product_dict_list(d):
    return (dict(zip(d, x)) for x in itertools.product(*d.values()))


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='main argument')

    parser.add_argument('-o', '--output', type=str,
                        default='./exp/',
                        help='Output path to exp result')

    parser.add_argument('-m', '--models', type=str, nargs='+',
                        default=['ridge', 'dtr', 'rf'],
                        help='Baseline models to use')

    parser.add_argument('--tuple', type=int, nargs='+',
                        default=(10, 7),
                        help='A tuple of integers')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

    parser.add_argument('--flag', action='store_true',
                        help='A boolean argument')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=2,
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

    #
    # args dump
    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

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

    splits = []
    splits.append(('train', x_train, y_train))
    splits.append(('valid', x_valid, y_valid))
    splits.append(('test', x_test, y_test))

    #
    # baselines
    res_logger.info(f'model/config\t\t\ttrain-mse\tvalid-mse\ttest-mse')
    for model in args.models:
        logging.info(f'\n\n\t\tConsidering {model} as baseline....')
        res_logger.info(f'{model}')

        #
        # hyperparameter grid search
        hyper_param_values = HYPERPARAMS[model]
        for config in cartesian_product_dict_list(hyper_param_values):
            logging.info(f'\t considering config:\n\t\t{config}')

            #
            # fixing random seed
            config.update({'random_state': rand_gen})
            config_str = ','.join(f'{k}:{v}' for k, v in config.items())

            #
            # build model
            reg = MODELS[model](**config)
            #
            # fit on train
            reg.fit(x_train, y_train)
            #
            # eval on all splits
            mse_list = []
            for s_name, s_x, s_y in splits:

                y_pred = reg.predict(s_x)
                rmse = np.sqrt(mean_squared_error(s_y, y_pred))
                logging.info(f'\tMSE on {s_name}:\t{rmse} ')
                mse_list.append(rmse)
            res_logger.info('{}\t\t\t{}'.format(config_str,
                                                '\t'.join(f'{m:.5f}' for m in mse_list)))
