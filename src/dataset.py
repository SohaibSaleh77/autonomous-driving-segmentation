"""Dataset management, augmentations, and preprocessing."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .config import CONFIG


def resize(image, mask, target_size=(256, 256)):
    """Resizes image and ground-truth mask identically.
    
    Notice the resampling methods. Images use BILINEAR to smooth out pixels.
    Masks use NEAREST. If we used Bilinear on a mask, we'd invent new class 
    integers that don't exist (e.g. blending class 1 and class 3 gives class 2).
    """
    image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
    mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)
    return image, mask


def _flip(tensor_img):
    """Flips 3D or 2D PyTorch tensor horizontally."""
    return torch.flip(tensor_img, dims=[-1])


def flip(image, mask, prob=0.5):
    """Synchronized random horizontal flip.
    
    If we flip the image, we MUST flip the mask the exact same way. Otherwise, 
    you're penalizing the model for predicting a car on the left when you 
    just moved the car to the right.
    """
    if np.random.rand() < prob:
        image = _flip(image)
        mask = _flip(mask)
    return image, mask


def normalize(img_tensor, mean=None, std=None):
    """Standardizes a float image tensor using ImageNet standard mean and std."""
    if mean is None:
        mean = CONFIG["normalize"]["mean"]
    if std is None:
        std = CONFIG["normalize"]["std"]
        
    # .view(3, 1, 1) reshapes the mean/std tensors so they broadcast correctly 
    # across the H x W of the image tensor.
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (img_tensor - mean_t) / std_t


def denormalize(img_tensor, mean=None, std=None):
    """Reverts standardized image tensor back to printable [0, 1] RGB tensor."""
    if mean is None:
        mean = CONFIG["normalize"]["mean"]
    if std is None:
        std = CONFIG["normalize"]["std"]
        
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    denorm = img_tensor.cpu() * std_t + mean_t
    return torch.clamp(denorm, 0.0, 1.0)


def to_rgb(mask_tensor, palette):
    """Converts 2D integer class mask [H, W] into an RGB visualization image [H, W, 3]."""
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.cpu().numpy().astype(np.int64)
    else:
        mask_np = mask_tensor.astype(np.int64)
        
    rgb_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(palette)):
        idx = (mask_np == class_id)
        rgb_mask[idx] = palette[class_id]
    return rgb_mask


class CamVid(Dataset):
    """Custom PyTorch Dataset for loading multi-class CamVid image-mask pairs."""

    def __init__(self, root_dir, target_size=(256, 256), augment=False):
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'labels')
        self.target_size = target_size
        self.augment = augment
        self.images = [f for f in sorted(os.listdir(self.img_dir)) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # CamVid specific quirk: masks have a '_P' suffix before the extension.
        mask_name = img_name.replace('.png', '_P.png')
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image, mask = resize(image, mask, self.target_size)

        # PIL image is H x W x C. PyTorch expects C x H x W. permute handles this.
        img_tensor = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        
        # Masks are just integer class IDs. No channel dimension needed.
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        # Sometimes CamVid has a 255 or 32 value for 'Void'. Clamp it so we don't 
        # index out of bounds during one_hot encoding later.
        mask_tensor = torch.clamp(mask_tensor, 0, 31)

        if self.augment:
            img_tensor, mask_tensor = flip(img_tensor, mask_tensor)

        img_tensor = normalize(img_tensor)

        return img_tensor, mask_tensor


def get_loaders(data_dir, batch_size=8, split_ratio=0.85, num_workers=2,
                pin_memory=True, target_size=(256, 256)):
    """Constructs training and validation DataLoaders.
    
    Instead of PyTorch's random_split, we manually slice the indices. This allows 
    us to pass the exact same dataset object to both train and val loaders, just 
    filtering the files they are allowed to see. 
    """
    train_dataset = CamVid(data_dir, target_size=target_size, augment=True)
    val_dataset = CamVid(data_dir, target_size=target_size, augment=False)
    
    dataset_size = len(train_dataset.images)
    indices = torch.randperm(dataset_size).tolist()
    train_size = int(split_ratio * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Overwrite the file lists so the datasets only look at their assigned images
    train_dataset.images = [train_dataset.images[i] for i in train_indices]
    val_dataset.images = [val_dataset.images[i] for i in val_indices]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, val_dataset
