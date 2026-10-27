"""LDC few-shot training (Li et al., "Logits DeConfusion with CLIP for Few-Shot Learning", CVPR 2025; LDC/ldc_train.py)
on the data package's splits, with a configurable logit scale.

The model, losses, optimizer, training loop and the val search of the fusion weight are LDC's own (ldc_train.Runner,
unmodified). --logit_scale replaces CLIP's frozen logit_scale (100), which multiplies the cosine similarities into
clip_logits. Unlike in CLIP-LoRA, ProLIP and SO, these logits are not only a cross-entropy temperature: they are the
base of the residual ada_logits, the target of the two L1 losses, and are fused with the MLP logits at test time, so
the scale also changes what the adapters learn to output. As in LDC, the few-shot train set is sampled with the global
random state after seeding, and the full val split selects the fusion weight.

    python train_ldc.py --dataset dtd --shots 16 --seed 1
    python train_ldc.py --dataset dtd --shots 16 --seed 1 --logit_scale 50 --backbone RN50
"""
import argparse
import csv
import importlib
import math
import os
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'LDC'))


class Tools:
    """Stand-in for alisuretool.Tools, the only two functions ldc_train uses."""

    @staticmethod
    def print(data, txt_path=None):
        print(data)

    @staticmethod
    def new_dir(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


sys.modules['alisuretool'] = types.ModuleType('alisuretool')
sys.modules['alisuretool.Tools'] = types.SimpleNamespace(Tools=Tools)

import ldc_train  # noqa: E402  (LDC/ldc_train.py)
from datasets.utils import Datum  # noqa: E402  (LDC's datasets)

from data import FEWSHOT, build_dataset, sample_fewshot  # noqa: E402

FIELDS = ['dataset', 'shots', 'seed', 'backbone', 'logit_scale', 'zero_shot_acc', 'mlp_acc', 'ada_acc', 'tot_acc',
          'ldc_acc']


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='C:/Users/asr/datasets')
    parser.add_argument('--dataset', type=str, default='dtd', choices=list(FEWSHOT))
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', choices=['RN50', 'ViT-B/16'])
    parser.add_argument('--logit_scale', default=100., type=float, help='replaces CLIP\'s logit scale')
    # Training arguments
    parser.add_argument('--lr', default=1e-3, type=float, help='LDC trains with lr / 10')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--train_epoch', default=50, type=int)
    parser.add_argument('--results', default=str(ROOT / 'results' / 'ldc' / 'results.csv'))
    args = parser.parse_args()
    # the rest of LDC's config (Config10Dataset), read by Runner and the model
    args.dataset_name, args.loss_lambda, args.fuse_type = args.dataset, [1.0, 1.0, 1.0, 1.0, 1.0], 2
    args.cache_dir = str(ROOT / 'LDC' / 'model' / 'clip')
    return args


def main():
    args = get_arguments()
    ldc_train.Config10Dataset.setup_seed(args.seed)

    dataset = build_dataset(args.dataset, args.root_path, shots=0)
    train = sample_fewshot(dataset.train, args.shots, random)  # LDC samples with the global random state
    datum = lambda items: [Datum(impath=path, label=label, classname=dataset.classnames[label]) for path, label in items]
    args.dataset = types.SimpleNamespace(train_x=datum(train), val=datum(dataset.val), test=datum(dataset.test),
                                         classnames=dataset.classnames,
                                         template=importlib.import_module(f'datasets.{args.dataset_name}').template)
    args.num_classes = len(dataset.classnames)

    runner = ldc_train.Runner(config=args)
    runner.clip_model.logit_scale.data.fill_(math.log(args.logit_scale))  # frozen: only adapters are trained
    print(f'Logit scale: {runner.clip_model.logit_scale.exp().item():g}')
    acc = runner.train()[0]

    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results.exists()
    with open(results, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(dict(zip(FIELDS, [args.dataset_name, args.shots, args.seed, args.backbone, args.logit_scale,
                                          *(round(acc[k], 2) for k in ('clip_logits', 'mlp_logits', 'ada_logits',
                                                                       'tot_logits', 'acc'))])))
    print('**** Final test accuracy: {:.2f}. ****'.format(acc['acc']))


if __name__ == '__main__':
    main()
