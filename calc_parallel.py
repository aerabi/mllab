import json
import os
import shlex
from subprocess import Popen


template = 'python3 workstation.py calc -t %d -i %d --raw --support-vector-machine --save calc/%d.json --seed 1'


def command(task_id, iterations=1000):
    return shlex.split(template % (task_id, iterations, task_id))


if __name__ == '__main__':
    # load OpenML-99
    with open('openml99.json') as openml99_file:
        openml99 = json.load(openml99_file)

    cwd = os.getcwd()
    processes = [Popen(command(task_id), cwd=cwd) for task_id in openml99]
    for process in processes:
        process.wait()
