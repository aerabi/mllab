from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np


def load_data(name_template='cluster_cuts/cutoff%02d.json', interval=range(5, 100, 5), func=np.median):
    j = []
    for i in interval:
        with open(name_template % i, 'r') as f:
            j += json.load(f)
    j = [x for x in j if 0.0 < x[1] < 1.0]
    m = defaultdict(list)
    for x in j:
        m[x[0]].append(x[1])

    X = [m[i] for i in interval]
    means = [func(x) for x in X]

    return X, means


if __name__ == '__main__':
    reihe = range(5, 100, 5)
    func = np.median
    X, means = load_data(interval=reihe, func=func)
    Y, moans = load_data(name_template='cluster_cuts/cutoff1d%02d.json', interval=reihe, func=func)

    base = 0.8138147527879279
    unif = 0.8091619789936539
    xx = list(range(1, len(reihe) + 1))

    plt.boxplot(X)
    plt.xticks(xx, map(str, list(reihe)))
    plt.plot(xx, means)
    plt.fill_between(xx, base, means, alpha=0.5)
    plt.plot(xx, moans, color='red')
    plt.plot(xx, [base] * len(reihe), color='purple')
    plt.plot(xx, [unif] * len(reihe), color='green')
    plt.grid()
    plt.show()
