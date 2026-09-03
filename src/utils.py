"""Environment diagnostics and dataset utilities."""
import os
import sys
import tarfile
import urllib.request

import numpy as np
import torch

from .config import CONFIG


def check_gpu():
    """First things first: what hardware are we working with?"""
    print("=" * 50)
    print(" SYSTEM & GPU ENVIRONMENT ")
    print("=" * 50)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"PyTorch Version   : {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available    : {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        # Get total VRAM, convert bytes to Gigabytes.
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Device        : {gpu_name}")
        print(f"Allocated VRAM    : {total_memory:.2f} GB")
        device = torch.device('cuda')
    else:
        print("WARNING: CUDA unavailable. Training on CPU will be slow.")
        device = torch.device('cpu')
        
    print("=" * 50 + "\n")
    return device


def count_params(model):
    """Calculates total and trainable parameters of a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_memory(model, input_size=(1, 3, 256, 256)):
    """Rough VRAM footprint estimation.
    
    This is a heuristic. It calculates the memory for the weights (params * 4 bytes 
    for float32) and multiplies by a fudge factor (2.5) to account for activations 
    and gradients. It won't be perfectly accurate, but it tells you if you're about 
    to OOM before you wait 10 minutes for a CUDA out of memory error.
    """
    total_params, _ = count_params(model)
    param_size_mb = (total_params * 4) / (1024 ** 2)  
    intermediate_acts = param_size_mb * 2.5 
    total_est_mb = param_size_mb + intermediate_acts
    return param_size_mb, total_est_mb


def get_classes():
    """Returns the 32 semantic class names for the CamVid dataset."""
    return [
        "Animal", "Archway", "Bicycle", "Building", "Car", "CartLorryFence",
        "Child", "Column_Pole", "Fence", "LaneBkgrnd", "LaneMarking", "SignSymbol",
        "Misc", "Motorcycle", "OtherMoving", "ParkingBlock", "Pedestrian", "Road",
        "RoadShoulder", "Sidewalk", "Sky", "BuildingBlock", "Bus", "SUT",
        "TrafficCone", "TrafficLight", "Train", "Tree", "Truck", "UtilityPole",
        "Vegetation", "Void"
    ]


def get_palette(seed=None):
    """Generates a deterministic 32-class RGB color map palette for visualization.
    
    If we don't set a seed, the colors will randomly change every time we run the script. 
    We also hardcode specific colors for classes like Road and Sky so they always look 
    natural in the overlay plots.
    """
    if seed is None:
        seed = CONFIG["visualization"]["palette_seed"]
        
    np.random.seed(seed)
    palette = np.random.randint(0, 255, size=(32, 3), dtype=np.uint8)
    
    # Hardcoding specific colors for visual consistency
    palette[17] = [128, 64, 128]  # Road
    palette[20] = [70, 130, 180]  # Sky
    palette[4]  = [0, 0, 142]     # Car
    palette[16] = [220, 20, 60]   # Pedestrian
    palette[27] = [107, 142, 35]  # Tree
    palette[3]  = [70, 70, 70]    # Building
    return palette


def download_data(url, archive_name, target_dir):
    """Downloads and extracts tar dataset archive."""
    if not os.path.exists(target_dir):
        print(f"[*] Downloading dataset from {url}...")
        archive_path = os.path.join("data", archive_name)
        urllib.request.urlretrieve(url, archive_path)
        
        print(f"[*] Extracting archive '{archive_name}'...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall("data")
        print("[+] Dataset downloaded and extracted successfully!")
    else:
        print(f"[+] Dataset directory '{target_dir}' already present.")
