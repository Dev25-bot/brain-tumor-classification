# src/data_preprocessing.py
"""Dataset loading, preprocessing and augmentation for the brain MRI images."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 4
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# ImageNet statistics: ToTensor() scales pixels to 0-1, then we standardise
# with the same mean/std the pretrained backbones were trained with.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation: rotation, horizontal/vertical flips, zoom, shifts and
# brightness/contrast changes. The crop keeps most of the scan so the tumour
# is not cut out of the image.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_device():
    """Pick the fastest available device: CUDA GPU, Apple Silicon GPU, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_data_root():
    train_dir = DATA_ROOT / "Training"
    test_dir = DATA_ROOT / "Testing"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected dataset folders at {train_dir} and {test_dir}. "
            "Keep the Kaggle-style Training/Testing folders inside the data directory."
        )


def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Return train/val/test loaders and class names.

    20% of the Training folder is held out for validation (fixed seed), and the
    Testing folder is only used for the final evaluation.
    """
    _validate_data_root()

    train_full_ds = datasets.ImageFolder(DATA_ROOT / "Training", transform=train_transform)
    val_full_ds = datasets.ImageFolder(DATA_ROOT / "Training", transform=val_test_transform)

    val_size = int(0.2 * len(train_full_ds))
    train_size = len(train_full_ds) - val_size

    # Use the same random split indices for both datasets
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train_full_ds)), [train_size, val_size], generator=generator
    )

    train_ds = torch.utils.data.Subset(train_full_ds, train_indices.indices)
    val_ds = torch.utils.data.Subset(val_full_ds, val_indices.indices)

    test_ds = datasets.ImageFolder(DATA_ROOT / "Testing", transform=val_test_transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    classes = train_full_ds.classes
    return train_loader, val_loader, test_loader, classes
