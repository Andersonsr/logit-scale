"""Reproduces the CLIP-LoRA few-shot experiments (Zanella & Ben Ayed, CVPRW 2024), optionally with other logit scales.

Runs train_lora.py with the paper's fixed hyperparameters (its defaults: ViT-B/16, r=2, alpha=1, q/k/v in both
encoders, lr 2e-4, batch 32, 500 * shots iterations) for every logit scale x dataset x shots x seed, and appends the
zero-shot and CLIP-LoRA test accuracies to a CSV. Finished runs are skipped, so an interrupted sweep resumes.

Usage:
    python run_clip_lora.py                                  # the paper: all 8 datasets, 1/2/4/8/16 shots, seeds 1/2/3
    python run_clip_lora.py --logit_scales 10 50 --datasets dtd eurosat --shots 16
    python run_clip_lora.py --summary                        # only print the tables of results so far
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from data import FEWSHOT

ROOT = Path(__file__).resolve().parent
DATASETS = list(FEWSHOT)
FIELDS = ['dataset', 'shots', 'seed', 'backbone', 'logit_scale', 'zero_shot_acc', 'lora_acc']


def read_results(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def run(args, dataset, shots, seed, logit_scale, log_path):
    cmd = [sys.executable, 'train_lora.py', '--root_path', args.root_path, '--dataset', dataset, '--shots', str(shots),
           '--seed', str(seed), '--backbone', args.backbone, '--logit_scale', str(logit_scale)]
    # torch and numpy in anaconda each ship an OpenMP runtime, which aborts the process without this
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', PYTHONUNBUFFERED='1')
    lines = []
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            log.write(line)
            lines.append(line)
            if 'accuracy' in line:
                print('   ', line.strip())
        proc.wait()
    out = ''.join(lines)
    zs = re.search(r"Zero-shot CLIP's test accuracy: (\d+\.\d+)", out)
    lora = re.search(r'Final test accuracy: (\d+\.\d+)', out)
    if proc.returncode != 0 or not (zs and lora):
        print(f'    FAILED (exit code {proc.returncode}), see {log_path}')
        return None
    return float(zs.group(1)), float(lora.group(1))


def summary(rows, backbone):
    accs = defaultdict(list)
    for r in rows:
        if r['backbone'] == backbone:
            accs[float(r['logit_scale']), r['dataset'], int(r['shots'])].append(
                (float(r['zero_shot_acc']), float(r['lora_acc'])))
    mean = lambda xs: sum(xs) / len(xs)
    for scale in sorted({k[0] for k in accs}):
        datasets = [d for d in DATASETS if (scale, d) in {k[:2] for k in accs}]
        shots = sorted({k[2] for k in accs if k[0] == scale})
        print(f'\nCLIP-LoRA test accuracy ({backbone}, logit scale {scale:g}), mean over seeds [number of seeds]')
        print(f'{"shots":>6}' + ''.join(f'{d:>16}' for d in datasets) + f'{"average":>16}')
        zs = [mean([a[0] for k, v in accs.items() if k[:2] == (scale, d) for a in v]) for d in datasets]
        print(f'{"0 (zs)":>6}' + ''.join(f'{a:>16.2f}' for a in zs) + f'{mean(zs):>16.2f}')
        for s in shots:
            cells, means = '', []
            for d in datasets:
                v = accs.get((scale, d, s))
                if v:
                    means.append(mean([a[1] for a in v]))
                    cells += f'{means[-1]:>12.2f} [{len(v)}]'
                else:
                    cells += f'{"-":>16}'
            avg = f'{mean(means):>16.2f}' if len(means) == len(datasets) else f'{"-":>16}'
            print(f'{s:>6}' + cells + avg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='C:/Users/asr/datasets')
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--logit_scales', nargs='+', type=float, default=[100], help='100 is the original CLIP-LoRA')
    parser.add_argument('--backbone', default='ViT-B/16')
    parser.add_argument('--out_dir', default=str(ROOT / 'results' / 'clip_lora'))
    parser.add_argument('--summary', action='store_true', help='only print the results tables')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_path = out_dir / 'results.csv'
    rows = read_results(results_path)
    if args.summary:
        summary(rows, args.backbone)
        return

    (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
    done = {(r['dataset'], int(r['shots']), int(r['seed']), r['backbone'], float(r['logit_scale'])) for r in rows}
    for scale in args.logit_scales:
        for dataset in args.datasets:
            for shots in args.shots:
                for seed in args.seeds:
                    if (dataset, shots, seed, args.backbone, scale) in done:
                        continue
                    print(f'logit scale {scale:g}, {dataset}, {shots} shots, seed {seed}')
                    tag = f'{args.backbone.replace("/", "-")}_scale{scale:g}_{dataset}_{shots}shots_seed{seed}'
                    accs = run(args, dataset, shots, seed, scale, out_dir / 'logs' / f'{tag}.log')
                    if accs is None:
                        continue
                    row = dict(zip(FIELDS, [dataset, shots, seed, args.backbone, scale, *accs]))
                    new_file = not results_path.exists()
                    with open(results_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDS)
                        if new_file:
                            writer.writeheader()
                        writer.writerow(row)
                    rows.append({k: str(v) for k, v in row.items()})
    summary(rows, args.backbone)


if __name__ == '__main__':
    main()
