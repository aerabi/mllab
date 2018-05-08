import json
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


if __name__ == '__main__':
    json.dump(main(), open('cutoffs.json', 'w'))
