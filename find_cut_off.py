import argparse
import json
import logging
import numpy as np
import os
import subprocess


def main(python_path, w_dir, iter=5, input_file='cluster/rawAllx1000.json', cutoffs=range(5, 100, 5), seeds=[1],
         task_ids=None):
    """
    create KDEs with different cutoffs, test them and return the result
    :param python_path: absolute path of python exec
    :param w_dir: absolute path of working directory
    :param iter: number of iterations on each dataset, in testing phase
    :param input_file: the name of the json input file
    :param cutoffs: iterable of cutoffs, in integer
    :param seeds: list of randoms seeds
    :return: f1 score for each of the cutoffs
    """
    features = [
        'estimator__bootstrap',
        'estimator__max_features',
        'estimator__min_samples_leaf',
        'estimator__min_samples_split',
    ]
    result = []
    if task_ids is None:
        task_ids = list(map(str, eval(open(os.path.join(w_dir, 'jobs.txt'), 'r').read())))

    if not os.path.exists('cuts'):
        os.makedirs('cuts')

    for i in cutoffs:
        kde_name = os.path.join(w_dir, 'cuts/cut%02d.kde' % i)
        arguments = [
            os.path.join(python_path, 'python'), os.path.join(w_dir, 'workstation.py'),
            'load', os.path.join(w_dir, input_file),
            '-H'] + features + [
            '-s', '%.2f' % (i / 100.0), '-S', kde_name,
        ]
        print(' '.join(arguments))
        r = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(r.stdout) > 2:
            print(r.stdout)
        if len(r.stderr) > 2:
            print('ERROR', r.stderr)
        for seed in seeds:
            output_file_name = os.path.join(w_dir, 'cuts/cut%02d-%d.o' % (i, seed))
            with open(output_file_name, 'w') as output_file:
                arguments = [
                    os.path.join(python_path, 'python'), os.path.join(w_dir, 'workstation.py'),
                    'calc', '--raw', '-i', str(iter),
                    '-C', '|'.join(features),
                    kde_name, '"lambda *x : (x[0] > 0.5, x[1], int(round(x[2])), int(round(x[3])))"',
                    '-t'] + task_ids + [
                    '--seed', str(seed),
                    '--save', os.path.join(w_dir, 'cuts/cut%02d-%d.json' % (i, seed))
                ]
                print(' '.join(arguments))
                r = subprocess.run(arguments, stdout=output_file, stderr=subprocess.PIPE)
            if len(r.stderr) > 2:
                print('ERROR', '\033[93m', r.stderr.decode('utf-8'), '\033[0m')
            with open(output_file_name, 'r') as f:
                o = f.read()
            score = o.split(' ')[-1].strip()
            try:
                result.append([i, float(score)])
            except ValueError:
                print('ERROR', '\033[93m', o, '\033[0m')
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
    logging.basicConfig(level=logging.DEBUG)
    cmd_parser = argparse.ArgumentParser('workstation')
    subparsers = cmd_parser.add_subparsers(dest='option')
    calc_parser = subparsers.add_parser('calc')
    calc_parser.add_argument('-i', '--iter', default=5, type=int, help='number of iterations on each of the datasets')
    calc_parser.add_argument('-s', '--seed', default=[1], type=int, nargs='+', help='random seeds')
    calc_parser.add_argument('-t', '--tasks', default=None, type=str, nargs='+', help='task IDs')
    calc_parser.add_argument('-c', '--cutoff', default=[5], type=int, nargs='+', help='cutoffs (in percent)')
    calc_parser.add_argument('-S', '--save', default='cutoff.json', type=str, help='output json file')
    calc_parser.add_argument('-p', '--python-path', default='/usr/bin', type=str, help='absolute path of python exec')
    calc_parser.add_argument('-w', '--working-dir', default='.', type=str, help='path of working directory')
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('-m', '--metric', default=['accuracy'], nargs='+',
                             choices=['accuracy', 'f1', 'recall', 'precision'])
    load_parser.add_argument('-f', '--function', default='np.mean', help='aggregator for each dataset, default=np.mean')
    args = cmd_parser.parse_args()
    if args.option == 'calc':
        results = main(python_path=args.python_path, w_dir=args.working_dir,
                       iter=args.iter, cutoffs=args.cutoff, seeds=args.seed, task_ids=args.tasks)
        json.dump(results, open(args.save, 'w'))
    else:
        import matplotlib.pyplot as plt
        colors = {'accuracy': 'blue', 'f1': 'red', 'precision': 'purple', 'recall': 'green'}
        aggregate_function = eval(args.function)
        legend = []
        for score_function in args.metric:
            all_scores = load(score_function=score_function, aggregate_function=aggregate_function)

            # plot
            ks, vs = [], []
            for k, v in all_scores.items():
                ks.append(k)
                vs.append(v)
                print(k, v)
            line, = plt.plot(ks, vs, label=score_function.capitalize(), color=colors[score_function])
            legend.append(line)
        plt.legend(handles=legend, loc=1)
        plt.xlabel('Cutoff')
        plt.ylabel('Scores')
        plt.savefig('cutoff-%s-%s.pdf' % ('-'.join(args.metric), aggregate_function.__name__))
