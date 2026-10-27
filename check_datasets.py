"""Checks that every dataset of the data package is present under a root folder and complete: the split files load,
the split sizes and number of classes are the expected ones, and every image listed exists and is not empty.

    python check_datasets.py --root_path /path/to/datasets
    python check_datasets.py --root_path /path/to/datasets --datasets dtd eurosat

Exits with status 1 if any dataset is missing or incomplete.
"""
import argparse
import os
import sys

from data import DATASETS, build_dataset

# name: (classes, train, val, test), with the CoOp splits for the few-shot datasets
EXPECTED = {
    'caltech101': (100, 4128, 1649, 2465),
    'oxford_pets': (37, 2944, 736, 3669),
    'stanford_cars': (196, 6509, 1635, 8041),
    'oxford_flowers': (102, 4093, 1633, 2463),
    'food101': (101, 50500, 20200, 30300),
    'fgvc': (100, 3334, 3333, 3333),
    'dtd': (47, 2820, 1128, 1692),
    'eurosat': (10, 13500, 5400, 8100),
    'ucf101': (101, 7639, 1898, 3783),
    'sun397': (397, 15880, 3970, 19850),
    'imagenetv2': (1000, None, None, 10000),
    'imagenet_sketch': (1000, None, None, 50889),
    'imagenet_a': (200, None, None, 7500),
    'imagenet_r': (200, None, None, 30000),
}


def check(name, root):
    """Returns a list of problems, empty if the dataset is complete."""
    try:
        dataset = build_dataset(name, root, shots=0)
    except (OSError, KeyError, ValueError) as e:
        return [f'cannot load the splits: {e}']
    problems = []
    classes, *sizes = EXPECTED[name]
    if len(dataset.classnames) != classes:
        problems.append(f'{len(dataset.classnames)} classes, expected {classes}')
    for split, expected in zip(('train', 'val', 'test'), sizes):
        items = getattr(dataset, split)
        if expected is not None and len(items) != expected:
            problems.append(f'{split}: {len(items)} images, expected {expected}')
        bad = [path for path, _ in items or [] if not os.path.isfile(path) or os.path.getsize(path) == 0]
        if bad:
            problems.append(f'{split}: {len(bad)} images missing or empty, e.g. {bad[0]}')
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='C:/Users/asr/datasets')
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    args = parser.parse_args()

    failed = []
    for name in args.datasets:
        problems = check(name, args.root_path)
        print(f"{name:16s} {'OK' if not problems else 'INCOMPLETE'}")
        for problem in problems:
            print(f'    {problem}')
        if problems:
            failed.append(name)

    print(f'\n{len(args.datasets) - len(failed)}/{len(args.datasets)} datasets complete.')
    if failed:
        print('Incomplete: ' + ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
