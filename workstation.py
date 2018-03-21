from __future__ import print_function

import argparse
import json
import pickle
import traceback

import sklearn
from sklearn.preprocessing import Imputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt

import openml
import numpy as np
import scipy
from scipy.stats import gaussian_kde

from optimizer import SmackDown
from optimizer.raw import Raw
from wrappers import GaussianKDEWrapper, CategoricalDistribution


class AlgorithmWorkstation:
    def __init__(self, sklearn_model):
        self._pipeline_ = None
        self._model_ = sklearn_model
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_pred = None

    def load_openml_task(self, task_id):
        task = openml.tasks.get_task(task_id)
        data_set = task.get_dataset()
        nominal_indices = data_set.get_features_by_type('nominal', [task.target_name])
        X, y = task.get_X_and_y()

        # encode
        # print(nominal_indices)
        # encoder = OneHotEncoder(categorical_features=np.array(nominal_indices))
        # encoder.fit(X)
        # X = encoder.transform(X)

        # split data
        train_ind, test_ind = task.get_train_test_split_indices()
        self.X_train, self.X_test = X[train_ind], X[test_ind]
        self.y_train, self.y_test = y[train_ind], y[test_ind]

        return self

    def random_search_optimize(self, configuration_space):
        optimizer = RandomizedSearchCV(self._model_, configuration_space, n_iter=100)
        optimizer.fit(self.X_train, self.y_train)
        self.y_pred = optimizer.predict(self.X_test)
        return self

    def fit(self):
        # train the model
        self._model_.fit(self.X_train, self.y_train)
        self.y_pred = self._model_.predict(self.X_test)
        return self

    def get_score(self, metric=accuracy_score):
        try:
            return metric(self.y_test, self.y_pred)
        except ValueError:
            return metric(self.y_test, self.y_pred, average='micro')


def calc(task_ids, iterations, save=False, random_forest=False, input_configuration_space=None, raw=False):
    scores, scores_optimized = list(), list()
    all_scores, all_additionals = list(), dict()
    steps = [('imputer', Imputer()),
             ('estimator', RandomForestClassifier())]
    classifier = sklearn.pipeline.Pipeline(steps=steps)

    def function_to_minimize(**params):
        classifier.set_params(**params)
        trained = AlgorithmWorkstation(classifier).load_openml_task(task_id).fit()
        additional_data = {'params': params, 'scores': {}}
        score_metrics = [f1_score, precision_score, accuracy_score, recall_score]
        for metric in score_metrics:
            additional_data['scores'][metric.__name__] = trained.get_score(metric)
        return 1 - trained.get_score(), additional_data

    for task_id in task_ids:
        try:
            print('Task %d' % task_id)
            if random_forest:
                # using random forest classifier
                configuration_space = {
                    'imputer__strategy': ['mean', 'median', 'most_frequent'],
                    'estimator__bootstrap': [True, False],
                    'estimator__max_features': scipy.stats.uniform(loc=0.1, scale=0.8),
                    'estimator__min_samples_leaf': list(range(1, 21)),
                    'estimator__min_samples_split': list(range(2, 21)),
                }

                for key, val in input_configuration_space.items():
                    configuration_space[key] = val

                scores.append(AlgorithmWorkstation(classifier).load_openml_task(task_id).fit().get_score())
                scores_optimized.append(
                    AlgorithmWorkstation(classifier)
                        .load_openml_task(task_id)
                        .random_search_optimize(configuration_space)
                        .get_score()
                )
            elif raw:
                # using random sampling
                print('using WWE Raw', end=' ')
                raw = Raw()
                raw.add('imputer__strategy', CategoricalDistribution(['mean', 'median', 'most_frequent']))
                raw.add('estimator__bootstrap', CategoricalDistribution([True, False]))
                raw.add('estimator__max_features', scipy.stats.uniform(loc=0.1, scale=0.8))
                raw.add('estimator__min_samples_leaf', CategoricalDistribution(list(range(1, 21))))
                raw.add('estimator__min_samples_split', CategoricalDistribution(list(range(2, 21))))

                for key, val in input_configuration_space.items():
                    if '|' in key:
                        raw.add_multi(key.split('|'), val[0], val[1])
                    else:
                        raw.add(key, val)

                scores = []
                additionals = []
                for i in range(iterations):
                    hyperparameters = raw.sample()
                    negative_score, additional = function_to_minimize(**hyperparameters)
                    scores.append(1 - negative_score)
                    additionals.append(additional)
                all_scores.append(scores)
                all_additionals[task_id] = additionals
                print('max score:', max(scores))
                scores_optimized.append(max(scores))
            else:
                params = {
                    'imputer__strategy': ('categorical', ['mean', 'median', 'most_frequent'], 'mean'),
                    'estimator__bootstrap': ('categorical', [True, False], True),
                    'estimator__max_features': ('real', [0.1, 0.9], 0.5),
                    'estimator__min_samples_leaf': ('integer', [1, 20], 1),
                    'estimator__min_samples_split': ('integer', [2, 20], 1),
                }

                smackdown = SmackDown(function_to_minimize, params, iterations)
                _, _, negative_scores_for_task, additionals = smackdown.minimize()
                positive_scores_for_task = [1 - score for score in negative_scores_for_task]
                all_scores.append(positive_scores_for_task)
                all_additionals[task_id] = additionals
                print(positive_scores_for_task)
        except Exception:
            print(traceback.format_exc())

    if save:
        with open(save, 'w') as output_file:
            json.dump(all_additionals, output_file)

    return scores, scores_optimized, all_scores


def plot(scores, scores_optimized, all_scores, task_ids, metric='accuracy', random_forest=False):
    if random_forest:
        print(np.mean(scores))
        print(np.mean(scores_optimized))

        data = [scores, scores_optimized]
        plt.boxplot(data)

        plt.xticks([1, 2], ['Default', 'Random Search'])
        plt.title('Random Forest Classifier Precision,\nwithout vs with Hyperparameter Optimization')
        plt.show()
    else:
        plt.boxplot(all_scores)

        plt.xticks([i + 1 for i in range(len(task_ids))], task_ids)
        plt.title('Random Forest Classifier %s,\nall values in grid search' % metric.capitalize())
        plt.xlabel('Datasets')
        plt.xticks(rotation=90)
        plt.ylabel(metric.capitalize())
        plt.show()


def plot_gaussian(kernel, a, b, hyperparameter, sample_percentage, metric):
    x_grid = np.linspace(a, b, 1000)
    pdf = kernel.evaluate(x_grid)
    plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    plt.xlim(a, b)
    plt.title('Gaussian Distribution\nbased on %d%% best %s scores' %
              (sample_percentage, metric))
    plt.xlabel(hyperparameter)
    plt.ylabel('distribution density')
    plt.show()


def main(args):
    if args.option == 'calc':
        configuration_space = {}
        if args.configuration is not None:
            for options in args.configuration:
                with open(options[1], 'rb') as input_file:
                    kdew = GaussianKDEWrapper(pickle.load(input_file), float(options[2]),
                                              float(options[3]), eval(options[4]))
                    configuration_space[options[0]] = kdew
            for options in args.configuration_multi:
                with open(options[1], 'rb') as input_file:
                    configuration_space[options[0]] = (pickle.load(input_file), eval(options[-1]))
        scores, scores_optimized, all_scores = calc(args.task_ids, args.iterations, args.save, args.random_forest,
                                                    configuration_space, args.raw)
        if len(scores_optimized) > 0:
            print('mean scores:', sum(scores_optimized) / len(scores_optimized))
        if args.plot:
            plot(scores, scores_optimized, all_scores, args.task_ids, 'accuracy', args.random_forest)

    elif args.option == 'load':
        data = json.load(args.file)
        metric = '%s_score' % args.metric.lower()

        if args.plot:
            keys = [int(key) for key in data.keys()]
            keys.sort()
            keys.reverse()
            task_ids = [str(key) for key in keys]
            all_scores = [[datum['scores'][metric]
                           for datum in data[task_id]] for task_id in task_ids]
            plot(None, None, all_scores, task_ids, args.metric, False)

        if args.sample > 0.0:
            all_samples = []
            for task_id, task_data in data.items():
                n_sample = int(args.sample * len(task_data))
                task_data.sort(key=lambda x: -x['scores'][metric])
                selected = task_data[:n_sample]
                if any(hp not in selected[0]['params'].keys() for hp in args.hyperparameter):
                    print('invalid hyperparameter, please select from among: %s'
                          % ', '.join(selected[0]['params'].keys()))
                    return
                all_samples += [[item['params'][hp] for hp in args.hyperparameter] for item in selected]
            all_samples_np = np.array(all_samples)
            if all_samples_np.shape[1] == 1:
                kde = gaussian_kde(all_samples_np.flatten())
                plot_gaussian(kde, float(args.bounds[0]), float(args.bounds[1]), args.hyperparameter,
                              int(args.sample * 100), args.metric)
            else:
                kde = gaussian_kde(all_samples_np.T)
            if args.save_sample:
                with open(args.save_sample, 'wb') as output:
                    pickle.dump(kde, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    default_task_ids = [125921, 125920, 14968, 9980, 9971, 9950, 9946, 3918, 3567, 53]
    cmd_parser = argparse.ArgumentParser('workstation')
    subparsers = cmd_parser.add_subparsers(dest='option')
    calc_parser = subparsers.add_parser('calc')
    calc_parser.add_argument('-t', '--task-ids', nargs='+', type=int, default=default_task_ids,
                             help='OpenML.org task IDs to do the experiment on, '
                                  'default is a set 10 small datasets')
    calc_parser.add_argument('-s', '--save',
                             help='file to save the huperparameter configuration spaces '
                                  'together with evaluation scores for each configuration')
    calc_parser.add_argument('--random-forest', action='store_true',
                             help='whether to use random forest for hyperparameter '
                                  'optimization; save unavailable for this option')
    calc_parser.add_argument('-r', '--raw', action='store_true',
                             help='simple random sampling with save support')
    calc_parser.add_argument('-i', '--iterations', type=int, default=100,
                             help='number of configurations, default = 100')
    calc_parser.add_argument('-p', '--plot', action='store_true', help='whether to plot the result')
    calc_parser.add_argument('-c', '--configuration', action='append', nargs=5,
                             help='load statistical distribution for a specific hyperparameter '
                                  'from file (random forest)',
                             metavar=('HYPERPARAMETER', 'FILE', 'LOWER_BOUND', 'UPPER_BOUND', 'ROUND'))
    calc_parser.add_argument('-C', '--configuration-multi', action='append', nargs=3,
                             help='load multivariate statistical distribution for multiple hyperparameters '
                                  'input as a single string, delimited by horizontal line "|"',
                             metavar=('HYPERPARAMETERS', 'FILE', 'OUTPUT_FILTER'))
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('file', type=argparse.FileType(),
                             help='file to load the hyperparameter configurations and '
                                  'their scores from')
    load_parser.add_argument('-p', '--plot', action='store_true', help='whether to plot the result')
    load_parser.add_argument('-m', '--metric', choices=['F1', 'PRECISION', 'RECALL', 'ACCURACY'],
                             default='ACCURACY', help='score metric')
    load_parser.add_argument('-s', '--sample', type=float, default=0.0,
                             help='sample percentage of data from each configuration space')
    load_parser.add_argument('-H', '--hyperparameter', type=str, nargs='+', default=['estimator__max_features'],
                             help='name(s) of hyperparameter(s) to sample Gaussian distribution for')
    load_parser.add_argument('-b', '--bounds', nargs=2, default=[0.0, 1.0],
                             help='hyperparameter values lower and upper bounds')
    load_parser.add_argument('-S', '--save-sample', help='file to save the Gaussian KDE')

    main(cmd_parser.parse_args())
