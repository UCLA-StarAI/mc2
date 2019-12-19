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
import logging
import csv

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def discretize_features(split, bins, dtype=np.int16):

    n_features = split.shape[1]
    discrete_split = np.zeros(split.shape, dtype=dtype)

    assert len(bins) == n_features

    for j in range(n_features):
        discrete_split[:, j] = np.digitize(split[:, j], bins=bins[j])

    return discrete_split


def preprocess_boston_house(valid_perc=0.2, bins='auto', save_orig_data=None, rand_gen=None):

    dataset_name = 'boston'
    #
    # load data
    from keras.datasets import boston_housing
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    logging.info(f'Loaded boston house data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    # n_train = len(x_train)
    # n_valid = int(np.floor(valid_perc * n_train))

    #
    # split data for validation
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=valid_perc, random_state=rand_gen)

    logging.info('')
    logging.info(f'Loaded boston house data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    if save_orig_data:
        data_output_path = os.path.join(save_orig_data, f'orig.{dataset_name}.pklz')
        with gzip.open(data_output_path, 'wb') as f:
            pickle.dump(((x_train, y_train), (x_valid, y_valid), (x_test, y_test)), f)

    #
    # i) create bins
    n_features = x_train.shape[1]
    if isinstance(bins, str):
        bin_method = bins
        bins = [np.histogram(x_train[:, j], bins=bin_method)[1] for j in range(n_features)]

    #
    # digitize features
    logging.info('discretizing features...')
    x_train = discretize_features(x_train, bins, dtype=np.int16)
    x_valid = discretize_features(x_valid, bins, dtype=np.int16)
    x_test = discretize_features(x_test, bins, dtype=np.int16)
    logging.info(f'\t{x_train[:20]}')

    if save_orig_data:
        data_output_path = os.path.join(save_orig_data, f'discrete.{dataset_name}.pklz')
        with gzip.open(data_output_path, 'wb') as f:
            pickle.dump(((x_train, y_train), (x_valid, y_valid), (x_test, y_test)), f)

    #
    # one-hot encode them
    logging.info('binarizing features...')
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
    ohe.fit(x_train)

    feature_map = []
    for h in range(len(ohe.categories_)):
        feature_map.extend([h for k in range(len(ohe.categories_[h]))])

    x_train = ohe.transform(x_train)
    x_valid = ohe.transform(x_valid)
    x_test = ohe.transform(x_test)
    logging.info(f'\t{x_train[:20]}')

    print(feature_map, len(feature_map))
    assert len(feature_map) == x_train.shape[1], (len(feature_map), x_train.shape[1])

    logging.info(f'\nProcessed boston house data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feature_map


def preprocess_continuous_discrete_data(data_path, dataset_name,
                                        features_to_drop,
                                        target_feature,
                                        binary_features,
                                        discrete_features,
                                        continuous_features,
                                        valid_perc=0.2, test_perc=0.3, bin_method='auto', max_bins=10, save_orig_data=None, rand_gen=None):

    import pandas as pd

    #
    # load data
    data_frame = pd.read_csv(data_path)
    logging.info(f'Loaded {dataset_name} dataframe with shape: {data_frame.shape}')

    print(data_frame.columns)
    #
    # get label, remove other features
    data_frame = data_frame.drop(features_to_drop, axis=1)

    #
    # train, valid, test split
    perm = rand_gen.permutation(data_frame.index)
    m = len(data_frame.index)
    train_perc = 1 - test_perc
    train_end = int(train_perc * m)
    valid_end = int(valid_perc * train_end) + train_end
    train = data_frame.ix[perm[:train_end]]
    valid = data_frame.ix[perm[train_end:valid_end]]
    test = data_frame.ix[perm[valid_end:]]

    y_train = np.array(train[target_feature].values)
    train = train.drop([target_feature], axis=1)
    y_valid = np.array(valid[target_feature].values)
    valid = valid.drop([target_feature], axis=1)
    y_test = np.array(test[target_feature].values)
    test = test.drop([target_feature], axis=1)

    #
    # discretize features
    #
    #
    # binary_features = ['waterfront']
    # discrete_features = ['date', 'bedrooms', 'view', 'condition', 'grade']
    # continuous_features = ['sqft_living', 'bathrooms', 'sqft_lot', 'floors', 'sqft_above', 'sqft_lot15',
    #                        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15']

    #
    # discretize continous features
    for c in continuous_features:
        bins = np.histogram(train[c], bins=bin_method)[1]

        if len(bins > max_bins):
            bins = np.histogram(train[c], bins=max_bins)[1]

        train[c] = np.digitize(train[c], bins=bins)
        valid[c] = np.digitize(valid[c], bins=bins)
        test[c] = np.digitize(test[c], bins=bins)

    logging.info(f'discretized {train.head()}')

    #
    # binarizing discrete features
    discrete_features += continuous_features

    feature_map = []
    for h, d in enumerate(discrete_features):
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
        ohe.fit(np.array(train[d]).reshape(-1, 1))

        # print('OHE cat', ohe.categories_)
        feature_map.extend([h for k in range(len(ohe.categories_[0]))])

        train_oe = ohe.transform(np.array(train[d]).reshape(-1, 1))
        valid_oe = ohe.transform(np.array(valid[d]).reshape(-1, 1))
        test_oe = ohe.transform(np.array(test[d]).reshape(-1, 1))
        for j in range(train_oe.shape[1]):

            train[f'{d}-{j}'] = train_oe[:, j]
            valid[f'{d}-{j}'] = valid_oe[:, j]
            test[f'{d}-{j}'] = test_oe[:, j]

        train = train.drop([d], axis=1)
        valid = valid.drop([d], axis=1)
        test = test.drop([d], axis=1)

    logging.info(f'binarized {train.head()}')

    logging.info(f'After binarization')

    x_train = np.array(train.values)
    x_valid = np.array(valid.values)
    x_test = np.array(test.values)

    print(feature_map, len(feature_map))
    assert (len(feature_map) + len(binary_features)
            ) == x_train.shape[1], (len(feature_map), x_train.shape[1])

    logging.info('')
    logging.info(f'Processed {dataset_name} data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feature_map


def preprocess_adult_data(train_data_path,
                          test_data_path,
                          valid_perc=0.2,
                          bin_method='auto', max_bins=10, save_orig_data=None, rand_gen=None):

    import pandas as pd

    target_feature = 'income'
    features_to_drop = []
    binary_features = []
    discrete_features = ['workclass', 'education', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'native-country']
    continuous_features = ['age', 'fnlwgt', 'education-num',
                           'capital-gain', 'capital-loss', 'hours-per-week']
    #
    # load data
    dataset_name = 'adult'
    train_data_frame = pd.read_csv(train_data_path)
    logging.info(f'Loaded train {dataset_name} dataframe with shape: {train_data_frame.shape}')

    test = pd.read_csv(test_data_path)
    logging.info(f'Loaded test {dataset_name} dataframe with shape: {test.shape}')

    print(train_data_frame.columns)
    print(train_data_frame.head())
    #
    # get label, remove other features
    train_data_frame = train_data_frame.drop(features_to_drop, axis=1)
    test = test.drop(features_to_drop, axis=1)

    #
    # drop missing values
    train_data_frame = train_data_frame[(train_data_frame != '?').all(axis=1)]
    test = test[(test != '?').all(axis=1)]

    #
    # train, valid, test split
    perm = rand_gen.permutation(train_data_frame.index)
    m = len(train_data_frame.index)
    train_perc = 1 - valid_perc
    train_end = int(train_perc * m)
    train = train_data_frame.ix[perm[:train_end]]
    valid = train_data_frame.ix[perm[train_end:]]

    #
    # getting labels
    print(train[target_feature])
    print(train[target_feature] == '<=50K')

    # train[train[target_feature] == '<=50K', target_feature] = 0
    train[target_feature] = train[target_feature].apply(lambda x: 0 if x == '<=50K' else 1)
    valid[target_feature] = valid[target_feature].apply(lambda x: 0 if x == '<=50K' else 1)
    test[target_feature] = test[target_feature].apply(lambda x: 0 if x == '<=50K.' else 1)

    # train[train[target_feature] == '>50K'][target_feature] = 1
    # valid[valid[target_feature] == '<=50K'][target_feature] = 0
    # valid[valid[target_feature] == '>50K'][target_feature] = 1
    # test[test[target_feature] == '<=50K'][target_feature] = 0
    # test[test[target_feature] == '>50K'][target_feature] = 1

    y_train = np.array(train[target_feature].values, dtype=np.int8)
    train = train.drop([target_feature], axis=1)
    y_valid = np.array(valid[target_feature].values, dtype=np.int8)
    valid = valid.drop([target_feature], axis=1)
    y_test = np.array(test[target_feature].values, dtype=np.int8)
    test = test.drop([target_feature], axis=1)

    #
    # discretize features
    #
    #
    # binary_features = ['waterfront']
    # discrete_features = ['date', 'bedrooms', 'view', 'condition', 'grade']
    # continuous_features = ['sqft_living', 'bathrooms', 'sqft_lot', 'floors', 'sqft_above', 'sqft_lot15',
    #                        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15']

    #
    # discretize continous features
    for c in continuous_features:
        bins = np.histogram(train[c], bins=bin_method)[1]

        if len(bins > max_bins):
            bins = np.histogram(train[c], bins=max_bins)[1]

        train[c] = np.digitize(train[c], bins=bins)
        valid[c] = np.digitize(valid[c], bins=bins)
        test[c] = np.digitize(test[c], bins=bins)

    logging.info(f'discretized {train.head()}')

    #
    # binarizing discrete features
    discrete_features += continuous_features
    feature_map = []
    for h, d in enumerate(discrete_features):
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
        ohe.fit(np.array(train[d]).reshape(-1, 1))

        feature_map.extend([h for k in range(len(ohe.categories_[0]))])

        train_oe = ohe.transform(np.array(train[d]).reshape(-1, 1))
        valid_oe = ohe.transform(np.array(valid[d]).reshape(-1, 1))
        test_oe = ohe.transform(np.array(test[d]).reshape(-1, 1))
        for j in range(train_oe.shape[1]):

            train[f'{d}-{j}'] = train_oe[:, j]
            valid[f'{d}-{j}'] = valid_oe[:, j]
            test[f'{d}-{j}'] = test_oe[:, j]

        train = train.drop([d], axis=1)
        valid = valid.drop([d], axis=1)
        test = test.drop([d], axis=1)

    logging.info(f'binarized {train.head()}')

    logging.info(f'After binarization')

    x_train = np.array(train.values)
    x_valid = np.array(valid.values)
    x_test = np.array(test.values)

    print(feature_map, len(feature_map))
    assert len(feature_map) == x_train.shape[1], (len(feature_map), x_train.shape[1])

    logging.info('')
    logging.info(f'Processed {dataset_name} data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feature_map


def preprocess_mnist35_data(data_path,
                            valid_perc=0.2,
                            sep=',',
                            save_orig_data=None, rand_gen=None):

    #
    # load train and valid splits as csv files
    lines = None
    train_path = os.path.join(data_path, 'train-3-5-images.txt')
    with open(train_path, 'r') as f:
        lines = list(csv.reader(f, delimiter=sep))
    x_train = np.array(lines, dtype=np.int8)

    lines = None
    train_label_path = os.path.join(data_path, 'train-3-5-labels.txt')
    with open(train_label_path, 'r') as f:
        lines = list(csv.reader(f, delimiter=sep))
    y_train = np.array(lines, dtype=np.int8).reshape(-1)

    test_path = os.path.join(data_path, 'test-3-5-images.txt')
    with open(test_path, 'r') as f:
        lines = list(csv.reader(f, delimiter=sep))
    x_test = np.array(lines, dtype=np.int8)

    lines = None
    test_label_path = os.path.join(data_path, 'test-3-5-labels.txt')
    with open(test_label_path, 'r') as f:
        lines = list(csv.reader(f, delimiter=sep))
    y_test = np.array(lines, dtype=np.int8).reshape(-1)

    #
    # splitting
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=valid_perc,
                                                          random_state=rand_gen)

    print(x_train[:20])

    print(x_valid[:20])
    #
    # already binarized feature map
    feature_map = [i for i in range(x_train.shape[1])]

    logging.info('')
    logging.info(f'Processed binary mnist data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feature_map


def preprocess_mnist_data(data_path,
                          valid_perc=0.2,
                          sep=',',
                          save_orig_data=None, rand_gen=None):
    import keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    #
    # reshaping
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    x_train = np.copy(x_train)
    x_test = np.copy(x_test)

    #
    # binarize according to mean
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    trainZ = x_train <= train_mean + 0.05 * train_std

    test_mean = np.mean(x_test, axis=0)
    test_std = np.std(x_test, axis=0)

    testZ = x_test <= test_mean + 0.05 * test_std

    nZ = np.logical_not(trainZ)
    tnZ = np.logical_not(testZ)

    x_train[trainZ] = 0
    x_train[nZ] = 1
    x_test[testZ] = 0
    x_test[tnZ] = 1

    #
    # splitting
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=valid_perc,
                                                          random_state=rand_gen)

    print(x_train[:20])

    print(x_valid[:20])
    #
    # already binarized feature map
    feature_map = [i for i in range(x_train.shape[1])]

    logging.info('')
    logging.info(f'Processed binary mnist data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feature_map


def preprocess_fmnist_data(data_path,
                           valid_perc=0.2,
                           sep=',',
                           save_orig_data=None, rand_gen=None):
    import keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    #
    # reshaping
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    x_train = np.copy(x_train)
    x_test = np.copy(x_test)

    #
    # binarize according to mean
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    trainZ = x_train <= train_mean + 0.05 * train_std

    test_mean = np.mean(x_test, axis=0)
    test_std = np.std(x_test, axis=0)

    testZ = x_test <= test_mean + 0.05 * test_std

    nZ = np.logical_not(trainZ)
    tnZ = np.logical_not(testZ)

    x_train[trainZ] = 0
    x_train[nZ] = 1
    x_test[testZ] = 0
    x_test[tnZ] = 1

    #
    # splitting
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=valid_perc,
                                                          random_state=rand_gen)

    print(x_train[:20])

    print(x_valid[:20])
    #
    # already binarized feature map
    feature_map = [i for i in range(x_train.shape[1])]

    logging.info('')
    logging.info(f'Processed fashion binary mnist data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feature_map


def preprocess_tcga_mutation_data(datapath,
                                  valid_perc=0.2, test_perc=0.3,
                                  feature_dtype=np.int8,
                                  sep=',', save_orig_data=None, rand_gen=None):

    lines = None
    with open(datapath, 'r') as f:
        lines = list(csv.reader(f, delimiter=sep))

    #
    # drop header
    header = lines[0]
    lines = lines[1:]

    if save_orig_data:
        data_output_path = os.path.join(save_orig_data, f'data.header')
        with open(data_output_path, 'w') as f:
            f.write(','.join(str(h).strip() for h in header))

    #
    # grab two last targets
    day_survival = np.array([float(m[-2]) for m in lines])
    vital_status = np.array([m[-1] for m in lines])

    #
    # make matrix
    X = []
    for m in lines:
        # print(m)
        x = [int(i) for i in m[1:-2] if i]
        X.append(x)
    X = np.array(X, dtype=feature_dtype)

    logging.info('\n Loaded mutation dataset with shapes:')
    logging.info(f'\t{X.shape}')

    #
    # splitting test part
    x_train, x_test, y_train, y_test = train_test_split(X, day_survival,
                                                        test_size=test_perc,
                                                        random_state=rand_gen)
    #
    # splitting validation
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=valid_perc, random_state=rand_gen)
    logging.info(f'\nProcessed mutation data with splits of shapes:')
    logging.info(f'\t\ttrain {x_train.shape} {y_train.shape}')
    logging.info(f'\t\tvalid {x_valid.shape} {y_valid.shape}')
    logging.info(f'\t\ttest  {x_test.shape} {y_test.shape}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='main argument')

    parser.add_argument("--dataset-name", type=str,
                        default='None',
                        help='Dataset name')

    parser.add_argument('-o', '--output', type=str,
                        default='./data/',
                        help='Output path for processed data')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--bins', type=str,
                        default='auto',
                        help='Binnins strategy')

    parser.add_argument('--valid-perc', type=float,
                        default=0.2,
                        help='Percentage of the training test for validation')

    parser.add_argument('--test-perc', type=float,
                        default=0.3,
                        help='Percentage of the training test for validation')

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

    out_path = args.output
    dataset_name = args.dataset_name
    os.makedirs(out_path, exist_ok=True)

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
    # processing
    print(args.dataset_name)
    if args.dataset_name == 'boston':

        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), f_map = preprocess_boston_house(
            valid_perc=args.valid_perc,
            bins=args.bins,
            save_orig_data=out_path,
            rand_gen=rand_gen)
    elif args.dataset_name == 'kings-county':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='kings-county',
                                                features_to_drop=['id'],
                                                target_feature='price',                                                                                                      binary_features=['waterfront'],
                                                discrete_features=[
                                                    'date', 'bedrooms', 'view', 'condition', 'grade'],
                                                continuous_features=['sqft_living', 'bathrooms', 'sqft_lot',
                                                                     'floors', 'sqft_above', 'sqft_lot15',
                                                                     'sqft_basement', 'yr_built',
                                                                     'yr_renovated', 'zipcode', 'lat',
                                                                     'long', 'sqft_living15'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)

    elif args.dataset_name == 'insurance':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),  f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='insurance',
                                                features_to_drop=[],
                                                target_feature='charges',
                                                binary_features=[],
                                                discrete_features=['children',
                                                                   'smoker', 'region', 'sex'],
                                                continuous_features=['age', 'bmi'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)
    elif args.dataset_name == 'delta-ailerons':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),  f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='delta-ailerons',
                                                features_to_drop=[],
                                                target_feature='Sa',
                                                binary_features=[],
                                                discrete_features=[],
                                                continuous_features=['RollRate', 'PitchRate',
                                                                     'currPitch', 'currRoll', 'diffRollRate'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)
    elif args.dataset_name == 'elevators':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),  f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='elevators',
                                                features_to_drop=[],
                                                target_feature='Goal',
                                                binary_features=[],
                                                discrete_features=[],
                                                continuous_features=['climbRate', 'Sgz', 'p', 'q',
                                                                     'curRoll', 'absRoll', 'diffClb',
                                                                     'diffRollRate',
                                                                     'diffDiffClb', 'SaTime1', 'SaTime2',
                                                                     'SaTime3', 'SaTime4', 'diffSaTime1',
                                                                     'diffSaTime2', 'diffSaTime3', 'diffSaTime4', 'Sa'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)
    elif args.dataset_name == 'kinematics':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),  f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='kinematics',
                                                features_to_drop=[],
                                                target_feature='y',
                                                binary_features=[],
                                                discrete_features=[],
                                                continuous_features=['theta1', 'theta2', 'theta3',
                                                                     'theta4', 'theta5', 'theta6', 'theta7',
                                                                     'theta8'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)

    elif args.dataset_name == 'compact':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),  f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='compact',
                                                features_to_drop=[],
                                                target_feature='usr',
                                                binary_features=[],
                                                discrete_features=[],
                                                continuous_features=['lread', 'lwrite', 'scall',
                                                                     'sread', 'swrite', 'fork',
                                                                     'exec', 'rchar', 'wchar',
                                                                     'pgout', 'ppgout', 'pgfree',
                                                                     'pgscan', 'atch', 'pgin', 'ppgin',
                                                                     'pflt', 'vflt', 'runqsz',
                                                                     'freemem', 'freeswap'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)

    elif args.dataset_name == 'appliances':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='appliances',
                                                features_to_drop=['date'],
                                                target_feature='Appliances',
                                                binary_features=[],
                                                discrete_features=[],
                                                continuous_features=['lights', 'T1', 'RH_1', 'T2', 'RH_2',
                                                                     'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5',
                                                                     'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
                                                                     'T9', 'RH_9', 'T_out', 'Press_mm_hg',
                                                                     'RH_out', 'Windspeed',
                                                                     'Visibility', 'Tdewpoint', 'rv1', 'rv2'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)

    elif args.dataset_name == 'abalone':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), f_map = \
            preprocess_continuous_discrete_data(args.dataset,
                                                dataset_name='abalone',
                                                features_to_drop=[],
                                                target_feature='Rings',
                                                binary_features=[],
                                                discrete_features=['Sex'],
                                                continuous_features=['Length', 'Diameter', 'Height', 'Whole weight',
                                                                     'Shucked weight', 'Viscera weight', 'Shell weight'],
                                                valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                save_orig_data=out_path, rand_gen=rand_gen)

    elif args.dataset_name == 'adult':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), f_map =  \
            preprocess_adult_data(train_data_path=os.path.join(args.dataset, 'adult.data'),
                                  test_data_path=os.path.join(args.dataset, 'adult.test'),
                                  valid_perc=args.valid_perc,
                                  max_bins=10, save_orig_data=out_path, rand_gen=rand_gen)
    elif args.dataset_name == 'mnist':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), f_map =  \
            preprocess_mnist_data(data_path=args.dataset,
                                  valid_perc=args.valid_perc,
                                  save_orig_data=out_path, rand_gen=rand_gen)
    elif args.dataset_name == 'fmnist':
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), f_map =  \
            preprocess_fmnist_data(data_path=args.dataset,
                                   valid_perc=args.valid_perc,
                                   save_orig_data=out_path, rand_gen=rand_gen)
    elif args.dataset_name in set(['BRCA']):

        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = preprocess_tcga_mutation_data(args.dataset,
                                                                                                 valid_perc=args.valid_perc, test_perc=args.test_perc,
                                                                                                 feature_dtype=np.int8,
                                                                                                 sep=',', save_orig_data=out_path, rand_gen=rand_gen)

    #
    # save feature_map
    f_map_path = os.path.join(out_path, f'fmap-{dataset_name}.pickle')
    with open(f_map_path, 'wb') as f:
        pickle.dump(f_map, f)

    #
    # save data, gzipping
    data_output_path = os.path.join(out_path, f'{dataset_name}.pklz')
    with gzip.open(data_output_path, 'wb') as f:
        pickle.dump(((x_train, y_train), (x_valid, y_valid), (x_test, y_test)), f)
    logging.info(f'saved data to {data_output_path}')
