from __future__ import print_function

import argparse
import json


def merge(files, output_name):
    merged = {}
    for file_name in files:
        with open(file_name, 'r') as f:
            loaded = json.load(f)
            for key, value in loaded.items():
                merged[key] = value
    with open(output_name, 'w') as output:
        json.dump(merged, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('merger')
    parser.add_argument('file', nargs='+', help='JSON files to merge')
    parser.add_argument('output', help='output file name')
    parsed = parser.parse_args()
    merge(parsed.file, parsed.output)
