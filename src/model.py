"""U-Net architecture blocks and model builder."""
import torch
import torch.nn as nn

from .config import CONFIG


class ConvBlock(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2 block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # bias=False because BatchNorm handles the bias shift. Saves params.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True saves a bit of memory
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Downsampling block: ConvBlock followed by MaxPool2d."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # We need the pre-pooled features for the skip connection!
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class Decoder(nn.Module):
    """Upsampling block: Transpose Conv, Concatenation skip connection, ConvBlock."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # in_channels for ConvBlock is out_channels * 2 because we concatenate 
        # the upsampled tensor with the skip connection tensor along the channel dim.
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip_features):
        x = self.up(x)
        # dim=1 is the channel dimension in PyTorch (N, C, H, W)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """Multi-Class U-Net Semantic Segmentation Architecture."""

    def __init__(self, in_channels=3, num_classes=32, feature_map_base=None):
        super().__init__()
        if feature_map_base is None:
            feature_map_base = CONFIG["model"]["feature_map_base"]

        self.enc1 = Encoder(in_channels, feature_map_base[0])
        self.enc2 = Encoder(feature_map_base[0], feature_map_base[1])
        self.enc3 = Encoder(feature_map_base[1], feature_map_base[2])
        self.enc4 = Encoder(feature_map_base[2], feature_map_base[3])

        self.bottleneck = ConvBlock(feature_map_base[3], feature_map_base[3] * 2)

        self.dec4 = Decoder(feature_map_base[3] * 2, feature_map_base[3])
        self.dec3 = Decoder(feature_map_base[3], feature_map_base[2])
        self.dec2 = Decoder(feature_map_base[2], feature_map_base[1])
        self.dec1 = Decoder(feature_map_base[1], feature_map_base[0])

        # 1x1 Conv to map the final feature vector to the number of classes
        self.final_conv = nn.Conv2d(feature_map_base[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path: capture features for skips, pass pooled to next layer
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        # Decoder path: upsample, merge with skip, pass to next layer
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.final_conv(d1)


def init_weights(model):
    """Applies He/Kaiming Normal weight initialization to Conv layers.
    
    ReLU activations kill gradients for negative outputs. Kaiming init scales 
    the weights based on fan-in to keep variance roughly 1, preventing 
    gradients from vanishing or exploding early in training.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
