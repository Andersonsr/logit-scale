"""CLIP-Adapter few-shot training (Gao et al., "CLIP-Adapter: Better Vision-Language Models with Feature Adapters";
CLIP-Adapter/clip_adapter.py) on the data package's loaders, with a configurable logit scale.

The adapter and the forward pass are clip_adapter.py's: a bottleneck MLP (dim -> dim/4 -> dim, ReLUs, no bias) on the
image features, mixed in as alpha * adapter(f) + (1 - alpha) * f, and a frozen text classifier with one prompt per
dataset. The repository only has this trainer, which runs inside CoOp's Dassl, so the rest is CoOp's setting, which
the paper says it follows: SGD (lr 2e-3, momentum 0.9, weight decay 5e-4), batch 32, one warmup epoch at a constant
lr of 1e-5 then a cosine schedule, 50/100/200 epochs for 1-2/4-8/16 shots, CoOp's augmentation, and train then val
sampled as in CoOp. The paper's "lr 1e-5" looks like CoOp's warmup lr; --lr sets it otherwise. The paper searches the
residual ratio alpha per dataset without reporting the values and the code fixes it to 0.2: --search picks it on the
few-shot val set instead.

--logit_scale is the inverse temperature multiplying the cosine similarities in the training loss; CLIP's learned
value, frozen in CLIP-Adapter, is 100. Test accuracy is an argmax, so the scale only acts through training.

    python train_clip_adapter.py --dataset dtd --shots 16 --seed 1
    python train_clip_adapter.py --dataset dtd --shots 16 --seed 1 --logit_scale 50 --search
"""
import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'CLIP-LoRA'))

import clip  # noqa: E402  (CLIP-LoRA's copy, the original OpenAI code)

from data import FEWSHOT, build_dataset, build_loader, sample_fewshot  # noqa: E402

# clip_adapter.py's CUSTOM_TEMPLATES
TEMPLATES = {
    'caltech101': 'a photo of a {}.',
    'oxford_pets': 'a photo of a {}, a type of pet.',
    'stanford_cars': 'a photo of a {}.',
    'oxford_flowers': 'a photo of a {}, a type of flower.',
    'food101': 'a photo of {}, a type of food.',
    'fgvc': 'a photo of a {}, a type of aircraft.',
    'dtd': '{} texture.',
    'eurosat': 'a centered satellite photo of {}.',
    'ucf101': 'a photo of a person doing {}.',
    'sun397': 'a photo of a {}.',
}

EPOCHS = {1: 50, 2: 50, 4: 100, 8: 100, 16: 200}  # CoOp's configs rn50_ep50, rn50_ep100, rn50
ALPHA_GRID = [0.2, 0.4, 0.6, 0.8]

FIELDS = ['dataset', 'shots', 'seed', 'backbone', 'logit_scale', 'alpha', 'epochs', 'zero_shot_acc', 'adapter_acc']


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='C:/Users/asr/datasets')
    parser.add_argument('--dataset', type=str, default='dtd', choices=list(FEWSHOT))
    parser.add_argument('--shots', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='RN50', choices=['RN50', 'RN101', 'ViT-B/16', 'ViT-B/32'])
    parser.add_argument('--logit_scale', default=100., type=float, help='inverse temperature of the training loss')
    parser.add_argument('--alpha', default=0.2, type=float, help='residual ratio of the adapter')
    parser.add_argument('--search', action='store_true', help='pick alpha on the few-shot validation set')
    # Training arguments
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--epochs', default=None, type=int, help="CoOp's 50/100/200 for 1-2/4-8/16 shots by default")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--results', default=str(ROOT / 'results' / 'clip_adapter' / 'results.csv'))
    args = parser.parse_args()
    args.epochs = args.epochs or EPOCHS[args.shots]
    return args


class Adapter(nn.Module):
    """clip_adapter.py's Adapter."""

    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


def logits(clip_model, adapter, images, text_weights, alpha):
    with torch.no_grad():  # the image encoder is frozen
        image_features = clip_model.encode_image(images.cuda())
    image_features = alpha * adapter(image_features) + (1 - alpha) * image_features
    return F.normalize(image_features, dim=-1) @ text_weights


@torch.no_grad()
def evaluate(clip_model, adapter, loader, text_weights, alpha):
    adapter.eval()
    correct, total = 0, 0
    for images, target in loader:
        correct += (logits(clip_model, adapter, images, text_weights, alpha).argmax(dim=1).cpu() == target).sum().item()
        total += len(target)
    return 100. * correct / total


def train(clip_model, train_loader, text_weights, alpha, args):
    adapter = Adapter(text_weights.shape[0]).cuda()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(args.epochs):
        # Dassl's constant warmup (1 epoch at 1e-5), then cosine annealing, updated once per epoch
        lr = 1e-5 if epoch == 0 else 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        for group in optimizer.param_groups:
            group['lr'] = lr
        adapter.train()
        total_loss = 0.
        for images, target in train_loader:
            loss = F.cross_entropy(args.logit_scale * logits(clip_model, adapter, images, text_weights, alpha),
                                   target.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f'Epoch {epoch + 1}/{args.epochs}, lr {lr:.2e}, loss {total_loss / len(train_loader):.4f}')
    return adapter


def main():
    args = get_arguments()
    torch.manual_seed(args.seed)

    clip_model, _ = clip.load(args.backbone)
    clip_model.float().eval()  # clip_adapter.py trains in fp32

    print('Preparing dataset.')
    dataset = build_dataset(args.dataset, args.root_path, shots=0)
    rng = random.Random(args.seed)  # CoOp samples train, then val
    train_items = sample_fewshot(dataset.train, args.shots, rng)
    val_items = sample_fewshot(dataset.val, min(args.shots, 4), rng)
    train_loader = build_loader(train_items, train=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = build_loader(dataset.test, num_workers=args.num_workers)

    with torch.no_grad():
        tokens = clip.tokenize([TEMPLATES[args.dataset].format(c) for c in dataset.classnames]).cuda()
        text_weights = F.normalize(clip_model.encode_text(tokens), dim=-1).t()

    zs_acc = evaluate(clip_model, nn.Identity(), test_loader, text_weights, 0.)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    alpha = args.alpha
    if args.search:
        val_loader = build_loader(val_items, num_workers=args.num_workers)
        best = -1.
        for grid_alpha in ALPHA_GRID:
            adapter = train(clip_model, train_loader, text_weights, grid_alpha, args)
            acc = evaluate(clip_model, adapter, val_loader, text_weights, grid_alpha)
            print(f'alpha {grid_alpha:g}: val accuracy {acc:.2f}')
            if acc > best:
                best, alpha = acc, grid_alpha
        print(f'Search: alpha {alpha:g} (val accuracy {best:.2f})')

    print(f'Logit scale: {args.logit_scale:g}, alpha {alpha:g}, {args.epochs} epochs')
    adapter = train(clip_model, train_loader, text_weights, alpha, args)
    acc = evaluate(clip_model, adapter, test_loader, text_weights, alpha)
    print('**** Final test accuracy: {:.2f}. ****'.format(acc))

    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results.exists()
    with open(results, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(dict(zip(FIELDS, [args.dataset, args.shots, args.seed, args.backbone, args.logit_scale, alpha,
                                          args.epochs, round(zs_acc, 2), round(acc, 2)])))


if __name__ == '__main__':
    main()
