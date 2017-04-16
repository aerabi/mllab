from __future__ import division, print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import pysmac


digits = datasets.load_digits()
X, y = digits.data, digits.target

X1, X2, y1, y2 = train_test_split(X, y)


def tree_classification_precision(criterion='gini', splitter='best', max_features=None,
                                  min_samples_split=2, min_samples_leaf=1):
    if max_features == 'None':
        max_features = None
    cls = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_features=max_features,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    cls.fit(X1, y1)
    pred = cls.predict(X2)
    return 1 - accuracy_score(y2, pred)


if __name__ == '__main__':
    print(1 - tree_classification_precision())
    print(1 - tree_classification_precision(criterion='entropy'))
    print(1 - tree_classification_precision(splitter='random'))
    print(1 - tree_classification_precision(criterion='entropy', splitter='random'))

    parameters = dict(
        criterion=('categorical', ['gini'], 'gini'),
        splitter=('categorical', ['best', 'random'], 'best'),
        max_features=('categorical', ['None', 'auto', 'sqrt', 'log2'], 'None'),
        min_samples_split=('integer', [2, 4], 2),
        min_samples_leaf=('integer', [1, 4], 1)
    )

    opt = pysmac.SMAC_optimizer()
    value, parameters = opt.minimize(tree_classification_precision, 1000, parameters)
    print((1 - value) * 100.0)
    print(parameters)
