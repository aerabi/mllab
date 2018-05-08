import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess


def main(iter=5, input_file='cluster/rawAllx1000.json', cutoffs=range(5, 100, 5)):
    """
    create KDEs with different cutoffs, test them and return the result
    :param iter: number of iterations on each dataset, in testing phase
    :param input_file: the name of the json input file
    :param cutoffs: iterable of cutoffs, in integer
    :return: f1 score for each of the cutoffs
    """
    features = [
        'estimator__bootstrap',
        'estimator__max_features',
        'estimator__min_samples_leaf',
        'estimator__min_samples_split',
    ]
    result = []
    for i in cutoffs:
        kde_name = 'cuts/cut%02d.kde' % i
        subprocess.run(['python', 'workstation.py', 'load', input_file,
                        '-H'] + features + [
                        '-s', '%.2f' % i, '-S', kde_name])
        output_file = open('cuts/cut%02d.o' % i, 'w')
        subprocess.run(['python', 'workstation.py', 'calc', '--raw', '-i', str(iter),
                        '-C', '|'.join(features),
                        # '-t'] + list(map(str, tasks)) + [
                        kde_name, '"lambda *xs : xs[0] > 0.5, x[1], int(round(x[2])), int(round(x[3]))"',
                        '--save', 'cuts/cut%02d.json' % i], stdout=output_file)
        output_file.close()
        with open('cuts/cut%02d.o' % i, 'r') as f:
            o = f.read()
        score = o.split(' ')[-1].strip()
        result.append([i, float(score)])
        print(i, score)
    return result


def load(cutoffs=range(5, 100, 5), aggregate_function=np.mean, score_function='accuracy'):
    all_scores = {}
    for i in cutoffs:
        scores = []
        with open('cuts/cut%02d.json' % i, 'r') as json_file:
            data = json.load(json_file)
            for key, datum in data.items():
                score = aggregate_function([x['scores']['%s_score' % score_function] for x in datum])
                scores.append(score)
        all_scores[i] = np.mean(scores)
    return all_scores


if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser('workstation')
    subparsers = cmd_parser.add_subparsers(dest='option')
    calc_parser = subparsers.add_parser('calc')
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('-m', '--metric', default='accuracy', choices=['accuracy', 'f1', 'recall', 'precision'])
    load_parser.add_argument('-f', '--function', default='np.mean', help='aggregator for each dataset, default=np.mean')
    args = cmd_parser.parse_args()
    if args.option == 'calc':
        json.dump(main(), open('cutoffs.json', 'w'))
    else:
        aggregate_function = eval(args.function)
        score_function = args.metric
        all_scores = load(score_function=score_function, aggregate_function=aggregate_function)

        # plot
        ks, vs = [], []
        for k, v in all_scores.items():
            ks.append(k)
            vs.append(v)
            print(k, v)
        plt.plot(ks, vs)
        plt.xlabel('Cutoff')
        plt.ylabel(score_function.capitalize())
        plt.savefig('cutoff-%s-%s.pdf' % (score_function, aggregate_function.__name__))
