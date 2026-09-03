"""Central configuration for the autonomous driving segmentation pipeline.

Keep your hyperparameters here. It keeps run.py clean and makes 
hyperparameter tuning a lot easier than hunting through 6 different files.
"""

CONFIG = {
    # ── Dataset / DataLoader ───────────────────────────────────────
    "dataset": {
        "target_size":  [256, 256],
        "batch_size":   8,
        "split_ratio":  0.85,
        "num_workers":  2,
        "pin_memory":   True,
    },

    # ── Normalization (ImageNet defaults) ──────────────────────────
    # We use ImageNet stats because even though we train from scratch,
    # standardizing inputs to a similar distribution helps gradient flow.
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },

    # ── Model architecture ─────────────────────────────────────────
    "model": {
        "in_channels":      3,
        "num_classes":      32,
        "feature_map_base": [64, 128, 256, 512],
    },

    # ── Loss function ──────────────────────────────────────────────
    "loss": {
        "num_classes":  32,
        "ce_weight":    0.5,
        "dice_weight":  0.5,
    },

    # ── Optimizer ──────────────────────────────────────────────────
    "optimizer": {
        "lr":            1e-3,
        "weight_decay":  1e-4,
    },

    # ── LR Scheduler ───────────────────────────────────────────────
    "scheduler": {
        "T_max": 10,
    },

    # ── Training loop ──────────────────────────────────────────────
    "training": {
        "num_epochs": 10,
    },

    # ── Visualization ──────────────────────────────────────────────
    "visualization": {
        "num_samples":  2,
        "palette_seed": 42,
    },
}
