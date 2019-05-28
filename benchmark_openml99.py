from __future__ import print_function

import json
import os
import shlex
import time
from subprocess import Popen, PIPE
from threading import Timer


template = 'python3 workstation.py calc -t %d -i %d --raw --support-vector-machine --save calc/%d.json --seed 1'
cwd = os.getcwd()


def command(task_id, iterations=1000):
    return shlex.split(template % (task_id, iterations, task_id))


def run(task_id, iterations, timeout_sec=60):
    proc = Popen(command(task_id, iterations), cwd=cwd, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


if __name__ == '__main__':
    # load OpenML-99
    with open('openml99.json') as openml_file:
        openml_study = json.load(openml_file)

    benchmark = []

    for task_id in openml_study:
        print('%-5d' % task_id, end='')
        row = [task_id, ]
        for i in [1, ]:
            then = time.time()
            run(task_id, iterations=i, timeout_sec=60)
            now = time.time()
            row.append(now - then)
            print('%.2f' % (now - then), end='')
        benchmark.append(row)
        print()

    json.dump(benchmark, open('openml99-benchmark-to.json', 'w'))
