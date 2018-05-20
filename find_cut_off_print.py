from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    j = []
    for i in range(5, 100, 5):
        with open('cuts/cutoff%02d.json' % i, 'r') as f:
            j += json.load(f)
    j = [x for x in j if 0.0 < x[1] < 1.0]
    m = defaultdict(list)
    for x in j:
        m[x[0]].append(x[1])

    X = [m[i] for i in range(5, 100, 5)]
    means = [np.median(x) for x in X]

    plt.boxplot(X)
    plt.xticks(list(range(1, 20)), map(str, list(range(5, 100, 5))))
    plt.plot(list(range(1, 20)), means)
    plt.grid()
    plt.show()
