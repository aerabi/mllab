from __future__ import print_function

import argparse
import json


def mean(l):
    return sum(l) / len(l)


def compare(base, result, score='accuracy', function=max):
    comparee = json.load(open(base, 'r'))
    compared = json.load(open(result, 'r'))
    keys = sorted(compared.keys(), key=int)
    col1s = []
    col2s = []
    for key in keys:
        col1 = function([i['scores']['%s_score' % score] for i in comparee[key]])
        col2 = function([i['scores']['%s_score' % score] for i in compared[key]])
        col1s.append(col1)
        col2s.append(col2)
        diff = col2 - col1
        sign = '+' if diff > 0 else ''
        print('%s & %.2f & %.2f & %s%.2f \\\\' % (key, col1 * 100, col2 * 100, sign, diff * 100))
    print('\\hline')
    col1 = sum(col1s) / len(col1s)
    col2 = sum(col2s) / len(col2s)
    diff = col2 - col1
    sign = '+' if diff > 0 else ''
    print('mean & %.2f & %.2f & %s%.2f' % (col1 * 100, col2 * 100, sign, diff * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('compare')
    parser.add_argument('base', help='baseline JSON file, to compare the results to')
    parser.add_argument('result', help='results JSON file')
    parser.add_argument('-f', '--function', default='max', help='aggregator for each dataset, default = max')
    parsed = parser.parse_args()
    compare(parsed.base, parsed.result, function=eval(parsed.function))
