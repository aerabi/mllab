from __future__ import division, print_function

import sklearn
from sklearn import datasets, tree

from optimizer.smackdown import SmackDown


class HyperparameterOptimizer:

    def __init__(self, data, target, classifier, parameters, metric):
        self.data = data
        self.target = target
        self.classifier = classifier
        self.parameters = parameters
        self.metric = metric

    @staticmethod
    def get_test_instance():
        digits = datasets.load_digits()
        data, target = digits.data, digits.target
        classifier = tree.DecisionTreeClassifier
        parameters = dict(
            criterion=('categorical', ['gini'], 'gini'),
            splitter=('categorical', ['best', 'random'], 'best'),
            max_features=('categorical', [None, 'auto', 'sqrt', 'log2'], None),
            min_samples_split=('integer', [2, 4], 2),
            min_samples_leaf=('integer', [1, 4], 1)
        )
        metric = sklearn.metrics.accuracy_score
        return HyperparameterOptimizer(data, target, classifier, parameters, metric)

    def __run_algorithm_per_configuration__(self, data_train, target_train, data_test, target_test):
        def __function_to_optimize__(**configuration):
            cls = self.classifier(**configuration)
            cls.fit(data_train, target_train)
            prediction = cls.predict(data_test)
            return 1 - self.metric(target_test, prediction)
        return __function_to_optimize__

    def run(self, iterations=20):
        # TODO: cross validation
        data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(self.data, self.target)
        function_to_optimize = self.__run_algorithm_per_configuration__(data_train, target_train, data_test, target_test)
        value, parameters = SmackDown(function_to_optimize, self.parameters, iterations).minimize()
        return 1 - value, parameters


if __name__ == '__main__':
    print(HyperparameterOptimizer.get_test_instance().run())


