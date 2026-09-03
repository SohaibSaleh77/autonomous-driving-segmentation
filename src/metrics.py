"""Loss functions and evaluation metric functions."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG


class Loss(nn.Module):
    """Combines Cross Entropy with Soft Multi-Class Dice Loss.
    
    Cross-Entropy alone suffers in segmentation because classes are highly imbalanced 
    (e.g. 80% road, 0.1% pedestrian). Dice loss optimizes the overlap directly, 
    handling imbalance better. Combining them gives stable gradients and good boundary adherence.
    """

    def __init__(self, num_classes=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        if num_classes is None:
            num_classes = CONFIG["loss"]["num_classes"]
            
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)

        probs = F.softmax(logits, dim=1)
        
        # Convert integer targets to one-hot encoding to match probs shape (N, C, H, W)
        # targets shape is (N, H, W). one_hot makes it (N, H, W, C). We permute to (N, C, H, W).
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3) # Sum over batch, height, and width. Keep classes separate.
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        # 1e-6 prevents division by zero for classes not present in the batch
        dice_score = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        dice_loss = 1.0 - torch.mean(dice_score)

        return (self.ce_weight * ce) + (self.dice_weight * dice_loss)


def pixel_acc(logits, targets):
    """Computes overall pixel accuracy across batch."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total


def per_class_iou(logits, targets, num_classes=None):
    """Computes Intersection-over-Union (IoU) vector for each individual class."""
    if num_classes is None:
        num_classes = CONFIG["model"]["num_classes"]
        
    preds = torch.argmax(logits, dim=1)
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        # If a class isn't in the ground truth OR the prediction, union is 0.
        # Append NaN so we know it wasn't evaluated, rather than artificially lowering the mean.
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)

    return iou_list


def mean_iou(logits, targets, num_classes=None):
    """Computes Mean Intersection-over-Union (mIoU) across present classes."""
    if num_classes is None:
        num_classes = CONFIG["model"]["num_classes"]
        
    ious = per_class_iou(logits, targets, num_classes)
    # Filter out the NaNs to get the true mean of evaluated classes
    valid_ious = [val for val in ious if not math.isnan(val)]
    return sum(valid_ious) / len(valid_ious) if len(valid_ious) > 0 else 0.0


def dice_coeff(logits, targets, num_classes=None):
    """Computes macro average Dice Score across present classes."""
    if num_classes is None:
        num_classes = CONFIG["model"]["num_classes"]
        
    preds = torch.argmax(logits, dim=1)
    dice_list = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).sum().item()
        total_pixels = pred_inds.sum().item() + target_inds.sum().item()

        if total_pixels == 0:
            continue
        dice = (2.0 * intersection) / total_pixels
        dice_list.append(dice)

    return sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0
