from __future__ import print_function

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def plot(file_name, feature1, feature2, metric, cutoff, smoothen=True):
    # load the data from the file
    with open(file_name) as data_file:
        data = json.load(data_file)
    rows = []
    datamap = {}
    for _, dataset in data.items():
        for datum in dataset:
            rows.append([
                datum['params'][feature1],
                datum['params'][feature2],
                datum['scores'][metric]
            ])
            datamap[(datum['params'][feature1], datum['params'][feature2])] = datum['scores'][metric]

    if smoothen:
        for i in range(-5, 15):
            for j in range(-15, 5):
                if (10 ** i, 10 ** j) not in datamap:
                    neighbours = [datamap[(10 ** (i + di), 10 ** (j + dj))]
                                  for di in [-1, 0, 1]
                                  for dj in [-1, 0, 1]
                                  if (10 ** (i + di), 10 ** (j + dj)) in datamap]
                    rows.append([10 ** i, 10 ** j, np.mean(neighbours)])

    if cutoff:
        n_rows_to_keep = int(len(rows) * cutoff)
        print('convert into NumPy array, sort, and cut top %d' % n_rows_to_keep)
        a = np.array(rows)
        top_rows = a[a[:, 2].argsort()[-n_rows_to_keep:]]
    else:
        a = np.array(rows)
        a_argsort = a[:, 2].argsort()
        percentile_cuts = []
        for co in np.arange(0.01, 1.01, 0.01):
            n_rows_to_keep = int(len(rows) * co)
            points = {(row[0], row[1]) for row in list(a[a_argsort[-n_rows_to_keep:]])}
            percentile_cuts += list(points)
        top_rows = np.array(percentile_cuts)

    # plot
    plt.hist2d(np.log10(top_rows[:, 0]), np.log10(top_rows[:, 1]), bins=[20, 20], range=[[-5, 15], [-15, 5]],
               cmap='hot')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('plot_2d_hist')
    parser.add_argument('file', help='calculated JSON file to plot')
    parser.add_argument('-x', default='estimator__C', help='first feature to plot')
    parser.add_argument('-y', default='estimator__gamma', help='second feature to plot')
    parser.add_argument('-m', '--metric', choices=['F1', 'PRECISION', 'RECALL', 'ACCURACY'],
                        default='ACCURACY', help='score metric to sort configurations according to')
    parser.add_argument('-c', '--cutoff', type=float, help='the ratio of best configurations to plot, [0., 1.]')
    parsed = parser.parse_args()
    plot(parsed.file, parsed.x, parsed.y, '%s_score' % parsed.metric.lower(), cutoff=parsed.cutoff)
