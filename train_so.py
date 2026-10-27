"""SO few-shot training (Mrabah et al., "Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation", ICCV 2025;
SO/main.py) on the data package's loaders, with a configurable logit scale.

The trained weights, the sparse optimizer and the training loop are SO's own (run_clip, unmodified). --logit_scale is
the inverse temperature multiplying the cosine similarities in the training loss; CLIP's learned value, and SO's
hard-coded one, is 100. Test accuracy is an argmax over cosine similarities, so the scale only acts through training.
SO stops when an epoch's mean loss falls below --early_stop_loss, and how soon that happens depends on the scale:
the number of iterations is saved with the results, and --early_stop_loss 0 always trains n_iters * shots iterations.

    python train_so.py --dataset dtd --shots 16 --seed 1
    python train_so.py --dataset dtd --shots 16 --seed 1 --logit_scale 50 --early_stop_loss 0
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'SO'))

import clip  # noqa: E402  (SO's copy)
import train as so_train  # noqa: E402  (SO/train.py)
from run_utils import set_random_seed  # noqa: E402

from data import FEWSHOT, TEMPLATE, build_dataset, build_loader  # noqa: E402

FIELDS = ['dataset', 'shots', 'seed', 'backbone', 'logit_scale', 'early_stop_loss', 'iterations', 'so_acc']


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='C:/Users/asr/datasets')
    parser.add_argument('--dataset', type=str, default='dtd', choices=list(FEWSHOT))
    parser.add_argument('--shots', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--logit_scale', default=100., type=float, help='inverse temperature of the training loss')
    # Location of trainable parameters
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=2000, type=int, help='the number of iterations is n_iters * shots')
    parser.add_argument('--early_stop_loss', default=0.01, type=float, help="Stop training early when epoch loss falls below this threshold.")
    parser.add_argument('--batch_size', default=32, type=int)
    # Optimizer arguments
    parser.add_argument('--s', default=0.99995, type=float, help='the sparsity ratio')
    parser.add_argument('--t', default=5, type=int, help='number of iterations before updating the sparsity support')
    parser.add_argument('--results', default=str(ROOT / 'results' / 'so' / 'results.csv'))
    args = parser.parse_args()
    args.eval_only, args.save_path = False, None  # read by run_clip
    return args


class CountingSO(so_train.SO):
    """SO's optimizer, counting its steps to record how many iterations early stopping left."""
    steps = 0

    def step(self):
        CountingSO.steps += 1
        return super().step()


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # SO prints emoji, which Windows' default encoding can't when piped
    args = get_arguments()
    set_random_seed(args.seed)

    clip_model, _ = clip.load(args.backbone, args.device)
    clip_model.eval()

    print('Preparing dataset.')
    dataset = build_dataset(args.dataset, args.root_path, args.shots, args.seed)
    dataset.template = [TEMPLATE]  # run_clip reads the prompt templates from the dataset
    test_loader = build_loader(dataset.test, num_workers=args.num_workers)
    train_loader = build_loader(dataset.train, train=True, batch_size=args.batch_size, num_workers=args.num_workers)

    # run_clip writes its result to ./results/final_results_*.txt and returns nothing: write it to our CSV instead
    def log_results(args, acc):
        results = Path(args.results)
        results.parent.mkdir(parents=True, exist_ok=True)
        new_file = not results.exists()
        with open(results, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(dict(zip(FIELDS, [args.dataset, args.shots, args.seed, args.backbone, args.logit_scale,
                                              args.early_stop_loss, CountingSO.steps, round(acc, 2)])))
        print(f'Trained for {CountingSO.steps} of {args.n_iters * args.shots} iterations.')

    so_train.log_results_to_csv = log_results
    so_train.SO = CountingSO

    print(f'Logit scale: {args.logit_scale}')
    so_train.run_clip(args, clip_model, args.logit_scale, dataset, train_loader, test_loader)


if __name__ == '__main__':
    main()
