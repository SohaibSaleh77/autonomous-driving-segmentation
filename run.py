"""End-to-end pipeline executor for Autonomous Driving Semantic Segmentation."""
import os
import time

import torch
import torch.optim as optim

from src.config import CONFIG
from src.utils import check_gpu, get_palette, get_classes, count_params, estimate_memory
from src.dataset import get_loaders
from src.model import UNet, init_weights
from src.metrics import Loss
from src.train import train_epoch, val_epoch, save
from src.visualize import plot_history, plot_samples, print_summary


def main():
    # Hardcoded directories as requested
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Environment Diagnostics
    device = check_gpu()

    # 2. Data Loaders & Color Map
    # Hardcoded "data/camvid" path
    train_loader, val_loader, val_dataset = get_loaders(
        "data/camvid",
        batch_size=CONFIG["dataset"]["batch_size"],
        split_ratio=CONFIG["dataset"]["split_ratio"],
        num_workers=CONFIG["dataset"]["num_workers"],
        pin_memory=CONFIG["dataset"]["pin_memory"],
        target_size=tuple(CONFIG["dataset"]["target_size"]),
    )
    palette = get_palette()
    class_names = get_classes()

    # 3. Instantiate Model
    model = UNet(
        in_channels=CONFIG["model"]["in_channels"],
        num_classes=CONFIG["model"]["num_classes"],
        feature_map_base=CONFIG["model"]["feature_map_base"],
    ).to(device)
    init_weights(model)

    total_p, train_p = count_params(model)
    p_mb, est_mb = estimate_memory(model)
    print(f"Model Summary      : {total_p:,} Total Params ({train_p:,} Trainable)")
    print(f"Est. Memory Footprint: ~{est_mb:.2f} MB VRAM")

    # 4. Loss & Optimizer Setup
    criterion = Loss(
        num_classes=CONFIG["loss"]["num_classes"],
        ce_weight=CONFIG["loss"]["ce_weight"],
        dice_weight=CONFIG["loss"]["dice_weight"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["optimizer"]["lr"],
        weight_decay=CONFIG["optimizer"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["scheduler"]["T_max"]
    )
    
    # Scaler for AMP. If on CPU, it's disabled.
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 5. Training Orchestration Loop
    num_epochs = CONFIG["training"]["num_epochs"]
    best_val_miou = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_miou': [], 'val_miou': [],
    }

    print("\n[+] Starting Pipeline Execution...\n")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_miou = val_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()
        elapsed = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - mIoU: {train_miou:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val mIoU: {val_miou:.4f}"
        )

        # Only save the model if it's better than what we've seen so far.
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
            })

    # 6. Post-Training Reporting
    print("\n[+] Training Complete! Generating Portfolio Diagnostics...")
    plot_history(history)
    plot_samples(
        model, val_dataset, palette, device,
        num_samples=CONFIG["visualization"]["num_samples"],
    )
    print_summary(model, val_loader, class_names, device)


if __name__ == "__main__":
    main()
