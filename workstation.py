from __future__ import print_function

import sklearn
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt

import openml
import numpy as np
import scipy


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
        return metric(self.y_test, self.y_pred)


def main(task_ids):
    scores, scores_optimized = list(), list()
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

    for task_id in task_ids:
        try:
            print('Task %d' % task_id)
            scores.append(AlgorithmWorkstation(classifier).load_openml_task(task_id).fit().get_score())
            scores_optimized.append(
                AlgorithmWorkstation(classifier)
                .load_openml_task(task_id)
                .random_search_optimize(configuration_space)
                .get_score()
            )
        except Exception as e:
            print(e)
    print(np.mean(scores))
    print(np.mean(scores_optimized))

    data = [scores, scores_optimized]
    plt.boxplot(data)

    plt.xticks([1, 2], ['Default', 'Random Search'])
    plt.title('Random Forest Classifier Precision,\nwithout vs with Hyperparameter Optimization')
    plt.show()


if __name__ == '__main__':
    task_ids = [125921, 125920, 14968, 9980, 9971, 9950, 9946, 3918, 3567, 53]
    main(task_ids)
