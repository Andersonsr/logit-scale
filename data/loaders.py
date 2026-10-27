"""PyTorch datasets and dataloaders over the (image path, label) lists of data.datasets, with CLIP's transforms."""
import torch
import torchvision.transforms as T
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# CLIP-LoRA's training augmentation
train_transform = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])

# the same as CLIP's own preprocess for 224px backbones
test_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])


class ImageList(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert('RGB')), label


def build_loader(items, train=False, batch_size=256, num_workers=8, transform=None):
    """Shuffled with augmentation when train, else in order with the test transform.

    Training loaders keep their workers between epochs: few-shot epochs are a few batches long, and on Windows
    starting the workers again for each one takes longer than the epoch itself.
    """
    transform = transform or (train_transform if train else test_transform)
    return torch.utils.data.DataLoader(ImageList(items, transform), batch_size=batch_size, shuffle=train,
                                       num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=train and num_workers > 0)
