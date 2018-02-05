from __future__ import print_function

import argparse
import json

import sklearn
from sklearn.preprocessing import Imputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt

import openml
import numpy as np
import scipy

from optimizer import SmackDown


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


def calc(task_ids, iterations, save=False, random_forest=False):
    scores, scores_optimized = list(), list()
    all_scores, all_additionals = list(), dict()
    steps = [('imputer', Imputer()),
             ('estimator', RandomForestClassifier())]
    classifier = sklearn.pipeline.Pipeline(steps=steps)

    configuration_space = {
        'imputer__strategy': ['mean', 'median', 'most_frequent'],
        'estimator__bootstrap': [True, False],
        'estimator__max_features': scipy.stats.uniform(loc=0.1, scale=0.8),
        'estimator__min_samples_leaf': list(range(1, 21)),
        'estimator__min_samples_split': list(range(2, 21)),
    }

    params = {
        'imputer__strategy': ('categorical', ['mean', 'median', 'most_frequent'], 'mean'),
        'estimator__bootstrap': ('categorical', [True, False], True),
        'estimator__max_features': ('real', [0.1, 0.9], 0.5),
        'estimator__min_samples_leaf': ('integer', list(range(1, 21)), 1),
        'estimator__min_samples_split': ('integer', list(range(2, 21)), 1),
    }

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
                scores.append(AlgorithmWorkstation(classifier).load_openml_task(task_id).fit().get_score())
                scores_optimized.append(
                    AlgorithmWorkstation(classifier)
                        .load_openml_task(task_id)
                        .random_search_optimize(configuration_space)
                        .get_score()
                )
            else:
                smackdown = SmackDown(function_to_minimize, params, iterations)
                _, _, negative_scores_for_task, additionals = smackdown.minimize()
                positive_scores_for_task = [1 - score for score in negative_scores_for_task]
                all_scores.append(positive_scores_for_task)
                all_additionals[task_id] = additionals
                print(positive_scores_for_task)
        except Exception as e:
            e.with_traceback()
            print(e)

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
        plt.ylabel(metric.capitalize())
        plt.show()


def main(args):
    if args.option == 'calc':
        scores, scores_optimized, all_scores = calc(args.task_ids, args.iterations,
                                                    args.save, args.random_forest)
        if args.plot:
            plot(scores, scores_optimized, all_scores, args.task_ids, 'accuracy', args.random_forest)

    elif args.option == 'load':
        data = json.load(args.file)
        keys = [int(key) for key in data.keys()]
        keys.sort()
        keys.reverse()
        task_ids = [str(key) for key in keys]
        all_scores = [[datum['scores']['%s_score' % args.metric.lower()]
                       for datum in data[task_id]] for task_id in task_ids]

        if args.plot:
            plot(None, None, all_scores, task_ids, args.metric, False)

if __name__ == '__main__':
    task_ids = [125921, 125920, 14968, 9980, 9971, 9950, 9946, 3918, 3567, 53]
    cmd_parser = argparse.ArgumentParser('workstation')
    subparsers = cmd_parser.add_subparsers(dest='option')
    calc_parser = subparsers.add_parser('calc')
    calc_parser.add_argument('-t', '--task-ids', nargs='+', type=int, default=task_ids,
                             help='OpenML.org task IDs to do the experiment on '
                                  '(only for recalculation)')
    calc_parser.add_argument('-s', '--save',
                             help='file to save the data after recalculation')
    calc_parser.add_argument('--random-forest', help='whether to use random forest for '
                                                     'hyperparameter optimization; save '
                                                     'unavailable for this option')
    calc_parser.add_argument('-i', '--iterations', type=int, default=100,
                             help='number of configurations')
    calc_parser.add_argument('-p', '--plot', action='store_true', help='whether to plot the result')
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('file', type=argparse.FileType(),
                             help='file to load the hyperparameter configurations and '
                                  'their scores from')
    load_parser.add_argument('-p', '--plot', action='store_true', help='whether to plot the result')
    load_parser.add_argument('-m', '--metric', choices=['F1', 'PRECISION', 'RECALL', 'ACCURACY'],
                             default='ACCURACY', help='score metric')

    main(cmd_parser.parse_args())
