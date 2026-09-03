# Autonomous Driving Semantic Segmentation

A PyTorch pipeline for multi-class semantic segmentation using a U-Net architecture. Built for learning and practice on the CamVid autonomous driving dataset.


Train
## Tech Stack

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Pillow

## Project Layout

```
autonomous_driving_segmentation/
├── pyproject.toml
├── .python-version
├── .gitignore
├── README.md
├── requirements.txt
├── download_data.py
├── run.py
├── data/               (ignored)
├── models/             (ignored)
├── reports/            (ignored)
└── src/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── dataset.py
    ├── model.py
    ├── metrics.py
    ├── train.py
    └── visualize.py
```

## How to Run (Colab / Local)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Download dataset
python download_data.py

# 3. Run the traiing pipeline (train -> evalute -> visuaize)
python run.py
```

## Methodology

- **Architecture:** U-Net with encoder-decoder skip connections.
- **Loss Function:** Cross-Entropy combined with Soft Dice Loss for stable convergence on imbalanced classes.
- **Optimization:** AdamW with Cosine Annealing learning rate scheduling.
- **Augmentations:** Synchronized horizontal flipping and joint image-mask resizing.
- **Mixed Precision:** PyTorch Automatic Mixed Precision (AMP) for optimized VRAM usage and speed.
- **Evaluation:** Pixel Accuracy, Mean Intersection-over-Union (mIoU), and per-class IoU breakdown.

## License

MIT
note..... the code is for educatinal purposes and not suitable for production.
CLASS NAME                | VALIDATION IoU      
==================================================
Animal                    | 0.0000
Archway                   | 0.0132
Bicycle                   | 0.3853
Building                  | 0.0000
Car                       | 0.8063
CartLorryFence            | 0.6081
Child                     | 0.0000
Column_Pole               | 0.0005
Fence                     | 0.0887
LaneBkgrnd                | 0.3956
LaneMarking               | 0.5938
SignSymbol                | 0.0000
Misc                      | 0.1025
Motorcycle                | 0.0000
OtherMoving               | 0.2327
ParkingBlock              | 0.0482
Pedestrian                | 0.2759
Road                      | 0.9154
RoadShoulder              | 0.0007
Sidewalk                  | 0.7228
Sky                       | 0.0340
BuildingBlock             | 0.9220
Bus                       | 0.0159
SUT                       | 0.0000
TrafficCone               | 0.3937
TrafficLight              | 0.0000
Train                     | 0.7611
Tree                      | 0.0826
Truck                     | 0.0000
UtilityPole               | 0.2638
Vegetation                | 0.2597
Void                      | 0.4425
==================================================

it shows code and network and logic work fine.
the dataset is small and imbalanced so peformamce is expected given this issues and no prevvous knowledge or  pretraining  . 
