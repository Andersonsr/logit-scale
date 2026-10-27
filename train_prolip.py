"""ProLIP few-shot training (Fahes et al., "CLIP's Visual Embedding Projector is a Few-shot Cornucopia") on the data
package's loaders, with a configurable logit scale.

As in ProLIP/methods/ProLIP.py, only CLIP's last visual projection is trained, full batch, for 300 epochs on
pre-projection features of 300 augmented views of the support set (one view per epoch), with the loss
    CE(logit_scale * cos(proj(x), text)) + lambda * ||W - W0||^2
Adam (eps 1e-4, fp16 weights as in ProLIP) with a cosine schedule. The defaults are ProLIP's validation-free setting
(lambda = 1/N); --search selects lr and lambda on an N-shot validation set instead (ProLIP's Table 1 setting).
CLIP's learned value of the logit scale, and ProLIP's hard-coded one, is 100. Several scales can be given: the
augmented views are encoded once and reused for each.

    python train_prolip.py --dataset dtd --shots 16 --seed 1
    python train_prolip.py --dataset dtd --shots 16 --seed 1 --logit_scales 100 50 10 --backbone RN50
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'CLIP-LoRA'))

import clip  # noqa: E402  (CLIP-LoRA's copy, the original OpenAI code)

from data import FEWSHOT, build_dataset, sample_fewshot  # noqa: E402
from data.loaders import CLIP_MEAN, CLIP_STD, ImageList, test_transform  # noqa: E402

# ProLIP's prompts (ProLIP/datasets/*.py)
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

# ProLIP's augmentation of the support set (milder crops than CLIP-LoRA's)
prolip_train_transform = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])

# ProLIP's grid for --search
LR_GRID = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
LAMBDA_GRID = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0]

FIELDS = ['dataset', 'shots', 'seed', 'backbone', 'logit_scale', 'lr', 'lambda', 'zero_shot_acc', 'prolip_acc']


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='C:/Users/asr/datasets')
    parser.add_argument('--dataset', type=str, default='dtd', choices=list(FEWSHOT))
    parser.add_argument('--shots', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', choices=['RN50', 'RN101', 'ViT-B/16', 'ViT-B/32'])
    parser.add_argument('--logit_scales', default=[100.], type=float, nargs='+',
                        help='inverse temperatures of the training loss, each trained separately')
    # Training arguments
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lambda_v', default='1/N', help="weight of ||W - W0||^2: a number, '1/N' or '1/N2'")
    parser.add_argument('--search', action='store_true', help='pick lr and lambda on an N-shot validation set')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--aug_views', default=300, type=int, help='augmented views of the support set')
    parser.add_argument('--batch_size', default=16, type=int, help='only for encoding the images')
    parser.add_argument('--results', default=str(ROOT / 'results' / 'prolip' / 'results.csv'))
    args = parser.parse_args()
    lambdas = {'1/N': 1 / args.shots, '1/N2': 1 / args.shots ** 2}
    args.lambda_v = lambdas[args.lambda_v] if args.lambda_v in lambdas else float(args.lambda_v)
    return args


def detach_projection(clip_model):
    """Makes encode_image return pre-projection features; returns the projection's weight and bias (None for ViT)."""
    visual = clip_model.visual
    if hasattr(visual, 'proj'):  # ViT: x @ proj
        w0 = visual.proj.detach().clone()
        visual.proj = None
        return w0, None
    c_proj = visual.attnpool.c_proj  # ResNet: the output projection of the attention pooling
    w0, b = c_proj.weight.detach().clone(), c_proj.bias.detach().clone()
    identity = nn.Linear(c_proj.in_features, c_proj.in_features).to(w0.device, w0.dtype)
    with torch.no_grad():
        identity.weight.copy_(torch.eye(c_proj.in_features))
        identity.bias.zero_()
    visual.attnpool.c_proj = identity
    return w0, b


def project(x, w, b):
    return x @ w if b is None else F.linear(x, w, b)


@torch.no_grad()
def text_classifier(clip_model, classnames, template):
    tokens = clip.tokenize([template.format(c) for c in classnames]).cuda()
    return F.normalize(clip_model.encode_text(tokens), dim=-1).t()


def loader(items, transform, args):
    return torch.utils.data.DataLoader(ImageList(items, transform), batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True,
                                       persistent_workers=args.num_workers > 0)


@torch.no_grad()
def encode(clip_model, data_loader):
    features, labels = [], []
    for images, target in data_loader:
        features.append(clip_model.encode_image(images.cuda()))
        labels.append(target)
    return torch.cat(features), torch.cat(labels).cuda()


def accuracy(features, labels, w, b, text_weights):
    pred = (F.normalize(project(features, w, b), dim=-1) @ text_weights).argmax(dim=1)
    return 100. * (pred == labels).float().mean().item()


def train(views, labels, text_weights, w0, b, logit_scale, lr, lambda_v, epochs):
    w = nn.Parameter(w0.clone())
    optimizer = torch.optim.Adam([w], lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for epoch in range(epochs):
        x = views[(epoch + 1) % len(views)].cuda()  # ProLIP's order: views 1, 2, ..., 0
        features = F.normalize(project(x, w, b), dim=-1)
        ce = F.cross_entropy(logit_scale * features @ text_weights, labels)
        loss = ce + lambda_v * (w - w0).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return w.detach(), ce.item()


def main():
    args = get_arguments()
    torch.manual_seed(args.seed)

    clip_model, _ = clip.load(args.backbone)
    clip_model.eval()

    print('Preparing dataset.')
    dataset = build_dataset(args.dataset, args.root_path, shots=0)
    rng = random.Random(args.seed)  # ProLIP samples train, then val
    train_items = sample_fewshot(dataset.train, args.shots, rng)
    val_items = sample_fewshot(dataset.val, args.shots, rng)

    text_weights = text_classifier(clip_model, dataset.classnames, TEMPLATES[args.dataset])
    w0, b = detach_projection(clip_model)

    print('Encoding the test set.')
    test_features, test_labels = encode(clip_model, loader(dataset.test, test_transform, args))
    zs_acc = accuracy(test_features, test_labels, w0, b, text_weights)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    print(f'Encoding {args.aug_views} augmented views of the {len(train_items)} training images.')
    train_loader = loader(train_items, prolip_train_transform, args)
    views, train_labels = [], None
    for _ in tqdm(range(args.aug_views)):
        features, train_labels = encode(clip_model, train_loader)
        views.append(features.cpu())
    if args.search:
        val_features, val_labels = encode(clip_model, loader(val_items, test_transform, args))

    for logit_scale in args.logit_scales:
        lr, lambda_v = args.lr, args.lambda_v
        if args.search:
            best = -1.
            for grid_lr in LR_GRID:
                for grid_lambda in LAMBDA_GRID:
                    w, _ = train(views, train_labels, text_weights, w0, b, logit_scale, grid_lr, grid_lambda, args.epochs)
                    acc = accuracy(val_features, val_labels, w, b, text_weights)
                    if acc > best:
                        best, lr, lambda_v = acc, grid_lr, grid_lambda
            print(f'Search: lr {lr:g}, lambda {lambda_v:g} (val accuracy {best:.2f})')

        w, ce = train(views, train_labels, text_weights, w0, b, logit_scale, lr, lambda_v, args.epochs)
        acc = accuracy(test_features, test_labels, w, b, text_weights)
        print(f'Logit scale {logit_scale:g}, lr {lr:g}, lambda {lambda_v:g}: final train CE {ce:.4f}')
        print('**** Final test accuracy: {:.2f}. ****\n'.format(acc))

        results = Path(args.results)
        results.parent.mkdir(parents=True, exist_ok=True)
        new_file = not results.exists()
        with open(results, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(dict(zip(FIELDS, [args.dataset, args.shots, args.seed, args.backbone, logit_scale, lr,
                                              lambda_v, round(zs_acc, 2), round(acc, 2)])))


if __name__ == '__main__':
    main()
