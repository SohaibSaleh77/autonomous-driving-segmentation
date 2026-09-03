"""Visualization and reporting utilities."""
import math

import numpy as np
import matplotlib.pyplot as plt
import torch

from .config import CONFIG
from .dataset import denormalize, to_rgb
from .metrics import per_class_iou


def plot_history(history):
    """Plots clean learning curve graphs for Loss, Accuracy, and mIoU."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    axes[0].set_title('Combined Segmentation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r--s', label='Val Acc')
    axes[1].set_title('Pixel Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(epochs, history['train_miou'], 'b-o', label='Train mIoU')
    axes[2].plot(epochs, history['val_miou'], 'r--s', label='Val mIoU')
    axes[2].set_title('Mean Intersection over Union (mIoU)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('mIoU Score')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("reports/training_history.png")
    plt.show()


def plot_samples(model, val_dataset, palette, device, num_samples=2):
    """Generates visual panel comparing Input, Ground Truth, Prediction, and Overlay."""
    model.eval()
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    
    # If num_samples is 1, axes is 1D. We expand it to 2D so the indexing loop doesn't break.
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices):
        img_tensor, true_mask = val_dataset[idx]

        with torch.no_grad():
            input_var = img_tensor.unsqueeze(0).to(device) # Add batch dimension
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(input_var)
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu()

        # Un-normalize the image so matplotlib renders it with correct colors
        denorm_img = denormalize(img_tensor).permute(1, 2, 0).numpy()
        rgb_true_mask = to_rgb(true_mask, palette)
        rgb_pred_mask = to_rgb(pred_mask, palette)

        # Blend the prediction and the original image 50/50
        overlay = (denorm_img * 0.5 + (rgb_pred_mask / 255.0) * 0.5)

        axes[i, 0].imshow(denorm_img)
        axes[i, 0].set_title("Input Dashboard Feed")
        axes[i, 1].imshow(rgb_true_mask)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 2].imshow(rgb_pred_mask)
        axes[i, 2].set_title("U-Net Predicted SegMap")
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Semantic Overlay")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("reports/prediction_samples.png")
    plt.show()


def print_summary(model, val_loader, class_names, device):
    """Prints a clean text breakdown of per-class IoU scores."""
    model.eval()
    accumulated_ious = np.zeros(len(class_names))
    counts = np.zeros(len(class_names))
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            ious = per_class_iou(logits, masks, len(class_names))

            # Accumulate IoUs and count how many batches had valid (non-NaN) IoUs
            for idx, val in enumerate(ious):
                if not math.isnan(val):
                    accumulated_ious[idx] += val
                    counts[idx] += 1

    print("\n" + "=" * 50)
    print(f"{'CLASS NAME':<25} | {'VALIDATION IoU':<20}")
    print("=" * 50)
    for i in range(len(class_names)):
        avg_iou = (accumulated_ious[i] / counts[i]) if counts[i] > 0 else 0.0
        print(f"{class_names[i]:<25} | {avg_iou:.4f}")
    print("=" * 50 + "\n")
