"""Checkpointing and training pipeline runners."""
import torch

from .config import CONFIG
from .metrics import pixel_acc, mean_iou


def save(state_dict, filepath="models/best_unet_camvid.pth"):
    """Saves model state and optimizer dicts to disk."""
    torch.save(state_dict, filepath)
    print(f"  [+] Saved new best checkpoint to '{filepath}'")


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Executes one training epoch with conditionally enabled PyTorch AMP.
    
    AMP (Automatic Mixed Precision) uses float16 for calculations where it can, 
    which drastically speeds up training and cuts VRAM usage in half on modern GPUs.
    The GradScaler prevents gradients from underflowing to zero in float16.
    """
    model.train()
    running_loss, running_acc, running_miou = 0.0, 0.0, 0.0
    
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_acc += pixel_acc(logits, masks)
        running_miou += mean_iou(logits, masks)

    n = len(dataloader)
    return running_loss / n, running_acc / n, running_miou / n


def val_epoch(model, dataloader, criterion, device):
    """Executes validation evaluation loop across dataset."""
    model.eval()
    running_loss, running_acc, running_miou = 0.0, 0.0, 0.0
    
    # no_grad saves memory and compute by not building the computation graph
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, masks)

            running_loss += loss.item()
            running_acc += pixel_acc(logits, masks)
            running_miou += mean_iou(logits, masks)

    n = len(dataloader)
    return running_loss / n, running_acc / n, running_miou / n
