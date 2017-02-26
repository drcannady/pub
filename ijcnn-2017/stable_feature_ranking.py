# Source code for the paper:
# Early Stablizing Feature Importance for TensorFlow Deep Neural Networks
#
# Copyright 2016 by Jeff Heaton, Steven McElwee, & James Cannady, Ph.D.
# Note: this version was tested with Google TensorFlow 0.80.
# (this is the version that was used to generate the published results)
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import tensorflow.contrib.learn as skflow
import pandas as pd
import os
import numpy as np
import codecs
import csv
import random

ENCODING = 'utf-8'
SHOW_RANKS = False
OPTIMIZER = "Adam"
LEARN_RATE = 0.01
BATCH_SIZE = 32

global_start = None
HIDDEN_UNITS = [200, 100, 50, 25]

path = "./data/"


# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.int32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)


def rank_diff(r1, r2):
    r1_temp = sorted(r1, key=lambda x: x[2])
    r2_temp = sorted(r2, key=lambda x: x[2])

    s = 0
    for i, j in zip(r1_temp, r2_temp):
        d = i[0] - j[0]
        s += d * d

    return np.sqrt(s)


def display_rank_line(l):
    return [x[1] for x in l]


def display_rank_num_line(l):
    return [x[0] for x in l]


class Ranking(object):
    def __init__(self, names):
        self.names = names

    def _normalize(self, x, y, impt):
        impt = impt / sum(impt)
        impt = list(zip(impt, self.names, range(x.shape[1])))
        impt.sort(key=lambda x: -x[0])
        return impt


class CorrelationCoefficientRank(Ranking):
    def __init__(self, names):
        super(CorrelationCoefficientRank, self).__init__(names)

    def rank(self, x, y, model=None):
        impt = []

        for i in range(x.shape[1]):
            c = abs(np.corrcoef(x[:, i], y[:, 0]))
            impt.append(abs(c[1, 0]))

        impt = impt / sum(impt)
        impt = list(zip(impt, self.names, range(x.shape[1])))
        impt.sort(key=lambda x: -x[0])

        return (impt)


class InputPerturbationRank(Ranking):
    def __init__(self, names):
        super(InputPerturbationRank, self).__init__(names)

    def _raw_rank(self, x, y, network):
        impt = np.zeros(x.shape[1])

        for i in range(x.shape[1]):
            hold = np.array(x[:, i])
            np.random.shuffle(x[:, i])
            pred = network.predict(x)
            rmse = metrics.mean_squared_error(y, pred)
            impt[i] = rmse
            x[:, i] = hold

        return impt

    def rank(self, x, y, network):
        impt = self._raw_rank(x, y, network)
        return self._normalize(x, y, impt)


class WeightRank(Ranking):
    # https://github.com/tensorflow/skflow/pull/111/files
    def __init__(self, names):
        super(WeightRank, self).__init__(names)

    def rank(self, x, y, network):
        weights = network.get_tensor_value('dnn/layer0/Linear/Matrix:0')
        weights = weights ** 2
        weights = np.sum(weights, axis=1)
        weights = np.sqrt(weights)
        weights = weights / sum(weights)

        result = self._normalize(x, y, weights)
        return result


class HybridRank(InputPerturbationRank):
    def __init__(self, names):
        super(HybridRank, self).__init__(names)

    def weight_vector(self, network):
        weights = network.get_tensor_value('dnn/layer0/Linear/Matrix:0')
        weights = weights ** 2
        weights = np.sum(weights, axis=1)
        weights = np.sqrt(weights)
        weights = weights / sum(weights)
        return weights

    def simple_vector(self, x, y):
        impt = []

        for i in range(x.shape[1]):
            c = np.corrcoef(x[:, i], y[:, 0])
            impt.append(abs(c[1, 0]))

        return np.array(impt)

    def rank(self, x, y, network):
        p_rank = self._raw_rank(x, y, network)
        w_rank = self.weight_vector(network)
        c_rank = self.simple_vector(x, y)

        d = (np.std(p_rank / sum(p_rank))) * 0.0

        impt = w_rank + (p_rank * d) + (c_rank * (1.0 - d))

        result = self._normalize(x, y, impt)
        return result


def rank_compare_experiment(x, y, names, ranker1, ranker2):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42)

    model = skflow.TensorFlowDNNRegressor(hidden_units=HIDDEN_UNITS, steps=50000,
                optimizer=OPTIMIZER, learning_rate=LEARN_RATE, batch_size = BATCH_SIZE)
    early_stop = skflow.monitors.ValidationMonitor(x_test, y_test,
                                                   early_stopping_rounds=200, print_steps=50)

    model.fit(x_train, y_train, monitor=early_stop)

    pred = model.predict(x_test)
    rmse = metrics.mean_squared_error(y_test, pred)
    print("RMSE: {}".format(rmse))

    print()
    print("*** {} ***".format(ranker1.__name__))
    l1 = ranker1(names).rank(x, y, model)

    for itm in l1:
        print(itm)

    print()
    print("*** {} ***".format(ranker2.__name__))
    l2 = ranker2(names).rank(x, y, model)
    display_rank_line(l2)

    for itm in l2:
        print(itm)

    print()
    print("Difference: {}".format(rank_diff(l1, l2)))


def rank_stability_experiment(x, y, names, ranker_class):
    # Step 1, see how far to train, using holdout set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=45)

    random.seed(42)
    model = skflow.TensorFlowDNNRegressor(hidden_units=HIDDEN_UNITS, steps=50000,
                optimizer=OPTIMIZER, learning_rate=LEARN_RATE, batch_size = BATCH_SIZE)
    early_stop = skflow.monitors.ValidationMonitor(x_test, y_test,
                                                   early_stopping_rounds=200, print_steps=50)

    model.fit(x_train, y_train, monitor=early_stop)

    pred = model.predict(x_test)
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print("RMSE: {}".format(score))
    baseline_rank = InputPerturbationRank(names).rank(x, y, model)
    steps_needed = int(model._monitor.min_loss_i)
    steps_inc = int(steps_needed / 20)

    # Step 2, rate stability up to that point

    ranker_base = InputPerturbationRank(names)
    ranker = ranker_class(names)

    with codecs.open("rank_stability.csv", "w", ENCODING) as fh:
        writer = csv.writer(fh)

        if SHOW_RANKS:
            writer.writerow(['steps', 'target_ptrb', 'target_hyb'] + [x for x in names])
        else:
            writer.writerow(['steps', 'target_ptrb', 'target_hyb'] + [x + 1 for x in range(len(names))])

        for i in range(20):
            steps = i * steps_inc
            random.seed(42)
            model = skflow.TensorFlowDNNRegressor(hidden_units=HIDDEN_UNITS, steps=steps,
                    optimizer=OPTIMIZER, learning_rate=LEARN_RATE, batch_size = BATCH_SIZE)
            model.fit(x, y, monitor=skflow.monitors.BaseMonitor(100000))

            r_base = ranker_base.rank(x, y, model)
            r_test = ranker.rank(x, y, model)
            d_base = rank_diff(baseline_rank, r_base)
            d_test = rank_diff(baseline_rank, r_test)

            if SHOW_RANKS:
                r_base.sort(key=lambda x: x[2])
                r_test.sort(key=lambda x: x[2])
                print("{}:{}:{}".format(steps, d_test, display_rank_num_line(r_test)))
                writer.writerow([steps, d_base, d_test] + display_rank_num_line(r_test) + [ranker.change_amount,
                                                                                           ranker.delta_weight])
            else:
                print("{}:{}:{}".format(steps, d_test, display_rank_line(r_test)))
                writer.writerow([steps, d_base, d_test] + display_rank_line(r_test))


def load_auto_mpg():
    filename_read = os.path.join(path, "auto-mpg.csv")
    df = pd.read_csv(filename_read, na_values=['NA', '?'])

    # create feature vector
    missing_median(df, 'horsepower')
    df.drop('name', 1, inplace=True)
    encode_numeric_zscore(df, 'horsepower')
    encode_numeric_zscore(df, 'weight')
    encode_numeric_zscore(df, 'cylinders')
    encode_numeric_zscore(df, 'displacement')
    encode_numeric_zscore(df, 'acceleration')
    encode_text_dummy(df, 'origin')
    x, y = to_xy(df, 'mpg')
    names = list(df.columns)
    names.remove('mpg')

    return names, x, y

def load_bupa():
    filename_read = os.path.join(path, "bupa.csv")
    df = pd.read_csv(filename_read, na_values=['NA', '?'])

    encode_numeric_zscore(df, 'mcv')
    encode_numeric_zscore(df, 'alkphos')
    encode_numeric_zscore(df, 'sgpt')
    encode_numeric_zscore(df, 'sgot')
    encode_numeric_zscore(df, 'gammagt')
    encode_numeric_zscore(df, 'drinks')
    #encode_numeric_zscore(df, 'selector')

    x, y = to_xy(df, 'selector')
    names = list(df.columns)
    names.remove('selector')

    return names, x, y

def load_wcbreast():
    filename = os.path.join(path,"wcbreast_wdbc.csv")
    df = pd.read_csv(filename,na_values=['NA','?'])

    # Encode feature vector
    df.drop('id',axis=1,inplace=True)
    encode_numeric_zscore(df,'mean_radius')
    encode_text_index(df,'mean_texture')
    encode_text_index(df,'mean_perimeter')
    encode_text_index(df,'mean_area')
    encode_text_index(df,'mean_smoothness')
    encode_text_index(df,'mean_compactness')
    encode_text_index(df,'mean_concavity')
    encode_text_index(df,'mean_concave_points')
    encode_text_index(df,'mean_symmetry')
    encode_text_index(df,'mean_fractal_dimension')
    encode_text_index(df,'se_radius')
    encode_text_index(df,'se_texture')
    encode_text_index(df,'se_perimeter')
    encode_text_index(df,'se_area')
    encode_text_index(df,'se_smoothness')
    encode_text_index(df,'se_compactness')
    encode_text_index(df,'se_concavity')
    encode_text_index(df,'se_concave_points')
    encode_text_index(df,'se_symmetry')
    encode_text_index(df,'se_fractal_dimension')
    encode_text_index(df,'worst_radius')
    encode_text_index(df,'worst_texture')
    encode_text_index(df,'worst_perimeter')
    encode_text_index(df,'worst_area')
    encode_text_index(df,'worst_smoothness')
    encode_text_index(df,'worst_compactness')
    encode_text_index(df,'worst_concavity')
    encode_text_index(df,'worst_concave_points')
    encode_text_index(df,'worst_symmetry')
    encode_text_index(df,'worst_fractal_dimension')
    encode_text_index(df,'diagnosis')

    x, y = to_xy(df, 'diagnosis')
    names = list(df.columns)
    names.remove('diagnosis')

    return names, x, y



def main():
    # Choose the dataset to use
    #names, x, y = load_auto_mpg()
    #names, x, y = load_bupa()
    #names, x, y = load_wcbreast()

    # Choose the experiment to run
    #rank_compare_experiment(x, y, names, InputPerturbationRank, CorrelationCoefficientRank)
    #rank_compare_experiment(x, y, names, WeightRank, InputPerturbationRank)
    rank_stability_experiment(x, y, names, HybridRank)


main()
