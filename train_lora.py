"""CLIP-LoRA few-shot training (CLIP-LoRA/main.py) on the data package's loaders, with a configurable logit scale.

The LoRA modules and the training loop are CLIP-LoRA's own (run_lora, unmodified). --logit_scale is the inverse
temperature multiplying the cosine similarities in the training loss; CLIP's learned value, and CLIP-LoRA's
hard-coded one, is 100. Test accuracy is an argmax over cosine similarities, so the scale only acts through training.

    python train_lora.py --dataset dtd --shots 16 --seed 1
    python train_lora.py --dataset dtd --shots 16 --seed 1 --logit_scale 50
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CLIP-LoRA'))

import clip  # noqa: E402  (CLIP-LoRA's copy)
from lora import run_lora  # noqa: E402
from run_utils import set_random_seed  # noqa: E402

from data import FEWSHOT, TEMPLATE, build_dataset, build_loader  # noqa: E402


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='C:/Users/asr/datasets')
    parser.add_argument('--dataset', type=str, default='dtd', choices=list(FEWSHOT))
    parser.add_argument('--shots', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--logit_scale', default=100, type=float, help='inverse temperature of the training loss')
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int, help='the number of iterations is n_iters * shots')
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA')
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    args = parser.parse_args()
    args.eval_only, args.save_path = False, None  # read by run_lora
    return args


def main():
    args = get_arguments()
    set_random_seed(args.seed)

    clip_model, _ = clip.load(args.backbone)
    clip_model.eval()

    print('Preparing dataset.')
    dataset = build_dataset(args.dataset, args.root_path, args.shots, args.seed)
    dataset.template = [TEMPLATE]  # run_lora reads the prompt templates from the dataset
    val_loader = build_loader(dataset.val, num_workers=args.num_workers)
    test_loader = build_loader(dataset.test, num_workers=args.num_workers)
    train_loader = build_loader(dataset.train, train=True, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f'Logit scale: {args.logit_scale}')
    run_lora(args, clip_model, args.logit_scale, dataset, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()
