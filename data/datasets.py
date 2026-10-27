"""Few-shot classification benchmarks of CoOp / CLIP-LoRA, read from the layout download_datasets.sh creates.

The 10 few-shot datasets use the CoOp train/val/test splits (split_zhou_*.json, or FGVC's own split files); the few-shot
train and val sets are sampled exactly as CLIP-LoRA does, so a given seed picks the same images. The ImageNet variants
only have a test set (they evaluate a model adapted on ImageNet); ImageNet-A and -R cover 200 of the 1000 classes and
are labelled 0..199, as in CoOp.
"""
import json
import os
import random
from dataclasses import dataclass

FEWSHOT = {  # name: (folder, image folder, split file)
    'caltech101': ('caltech-101', '101_ObjectCategories', 'split_zhou_Caltech101.json'),
    'oxford_pets': ('oxford_pets', 'images', 'split_zhou_OxfordPets.json'),
    'stanford_cars': ('stanford_cars', '', 'split_zhou_StanfordCars.json'),  # split paths include cars_train/, cars_test/
    'oxford_flowers': ('oxford_flowers', 'jpg', 'split_zhou_OxfordFlowers.json'),
    'food101': ('food-101', 'images', 'split_zhou_Food101.json'),
    'fgvc': ('fgvc_aircraft', 'images', None),
    'dtd': ('dtd', 'images', 'split_zhou_DescribableTextures.json'),
    'eurosat': ('eurosat', '2750', 'split_zhou_EuroSAT.json'),
    'ucf101': ('ucf101', 'UCF-101-midframes', 'split_zhou_UCF101.json'),
    'sun397': ('sun397', 'SUN397', 'split_zhou_SUN397.json'),
}
IMAGENET_VARIANTS = {  # name: (folder, image folder)
    'imagenetv2': ('imagenetv2', 'imagenetv2-matched-frequency-format-val'),
    'imagenet_sketch': ('imagenet-sketch', 'images'),
    'imagenet_a': ('imagenet-adversarial', 'imagenet-a'),
    'imagenet_r': ('imagenet-rendition', 'imagenet-r'),
}
DATASETS = list(FEWSHOT) + list(IMAGENET_VARIANTS)

TEMPLATE = 'a photo of a {}.'  # the single prompt CLIP-LoRA uses for every dataset


@dataclass
class Dataset:
    """Lists of (image path, label); train and val are None for the ImageNet variants."""
    name: str
    classnames: list
    train: list
    val: list
    test: list


def build_dataset(name, root, shots=16, seed=1):
    """`shots` images per class for train and min(shots, 4) for val; shots < 1 keeps the full splits."""
    if name in IMAGENET_VARIANTS:
        return _imagenet_variant(name, root)
    folder, image_folder, split_file = FEWSHOT[name]
    folder, image_dir = os.path.join(root, folder), os.path.join(root, folder, image_folder)
    if split_file is None:
        train, val, test, classnames = _fgvc(folder, image_dir)
    else:
        with open(os.path.join(folder, split_file)) as f:
            split = json.load(f)
        train, val, test = ([(os.path.join(image_dir, path), label) for path, label, _ in split[s]]
                            for s in ('train', 'val', 'test'))
        names = {label: cname for _, label, cname in split['train']}
        classnames = [names[label].replace('_', ' ') for label in sorted(names)]
    if shots >= 1:
        rng = random.Random(seed)  # CLIP-LoRA samples val, then train
        val = sample_fewshot(val, min(shots, 4), rng)
        train = sample_fewshot(train, shots, rng)
    return Dataset(name, classnames, train, val, test)


def sample_fewshot(items, shots, rng):
    """`shots` items per label (with repetition if a label has fewer), as CoOp samples them."""
    by_label = {}
    for item in items:
        by_label.setdefault(item[1], []).append(item)
    out = []
    for group in by_label.values():
        out += rng.sample(group, shots) if len(group) >= shots else rng.choices(group, k=shots)
    return out


def _fgvc(folder, image_dir):
    with open(os.path.join(folder, 'variants.txt')) as f:
        classnames = [line.strip() for line in f]
    label = {c: i for i, c in enumerate(classnames)}
    splits = []
    for s in ('train', 'val', 'test'):
        with open(os.path.join(folder, f'images_variant_{s}.txt')) as f:
            lines = [line.strip().split(' ', 1) for line in f]
        splits.append([(os.path.join(image_dir, im + '.jpg'), label[cname]) for im, cname in lines])
    return (*splits, classnames)


def _imagenet_variant(name, root):
    folder, image_folder = IMAGENET_VARIANTS[name]
    image_dir = os.path.join(root, folder, image_folder)
    with open(os.path.join(root, folder, 'classnames.txt')) as f:
        wnids, names = zip(*(line.strip().split(' ', 1) for line in f))
    if name == 'imagenetv2':  # class folders are named 0..999
        folders = [str(i) for i in range(len(wnids))]
        classnames = list(names)
    else:  # class folders are WordNet ids
        folders = sorted(d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d)))
        classnames = [names[wnids.index(d)] for d in folders]
    test = []
    for label, d in enumerate(folders):
        class_dir = os.path.join(image_dir, d)
        test += [(os.path.join(class_dir, im), label) for im in sorted(os.listdir(class_dir))]
    return Dataset(name, classnames, None, None, test)
