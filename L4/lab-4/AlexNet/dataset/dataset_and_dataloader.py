import os
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader




def get_train_transforms() -> T.Compose:
    """
    Build training preprocessing pipeline with data augmentation.
    """
    transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop((227, 227)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform






def get_eval_transforms() -> T.Compose:
    """
    Build validation/test preprocessing pipeline.
    No data augmentation here.
    """
    transform = T.Compose([
        T.Resize((227, 227)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform






def build_datasets(data_root: str):
    """
    Build TinyImageNet train/val datasets.

    Expected structure:
    data_root/
        tiny-imagenet-200/
            train/
            val/
    """
    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    train_path = os.path.join(data_root, "tiny-imagenet-200", "train")
    val_path = os.path.join(data_root, "tiny-imagenet-200", "val")

    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_path, transform=eval_transform)

    return train_dataset, val_dataset






def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Build TinyImageNet train/val dataloaders.
    """
    train_dataset, val_dataset = build_datasets(data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader