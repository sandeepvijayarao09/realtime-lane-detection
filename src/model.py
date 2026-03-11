"""
LaneNet: Real-time Lane Detection Network

Architecture: Encoder-Decoder CNN with skip connections
- Backbone: EfficientNet or MobileNet from torchvision
- Segmentation head: Binary lane mask prediction
- Embedding head: Instance-level lane representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connections."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LaneNet(nn.Module):
    """
    LaneNet for real-time lane detection.

    Args:
        num_classes: Number of lane instances (default: 1 for binary segmentation)
        backbone: 'efficientnet' or 'mobilenet' (default: 'efficientnet')
        pretrained: Use pretrained backbone weights (default: True)
        embedding_dim: Dimension of instance embedding (default: 4)
    """

    def __init__(self, num_classes: int = 1, backbone: str = 'efficientnet',
                 pretrained: bool = True, embedding_dim: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Load pretrained backbone
        if backbone == 'efficientnet':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            # EfficientNet-B0 has these channel sizes at different depths
            self.backbone_channels = [16, 24, 40, 80, 320]
        elif backbone == 'mobilenet':
            base_model = models.mobilenet_v2(pretrained=pretrained)
            # MobileNetV2 has different channel organization
            self.backbone_channels = [16, 24, 32, 96, 320]
        else:
            raise ValueError(f"Backbone must be 'efficientnet' or 'mobilenet', got {backbone}")

        # Extract encoder layers from backbone
        self._setup_encoder(base_model, backbone)

        # Decoder: progressively upsample and add skip connections
        self.decoder4 = DecoderBlock(self.backbone_channels[4], self.backbone_channels[3], 256)
        self.decoder3 = DecoderBlock(256, self.backbone_channels[2], 128)
        self.decoder2 = DecoderBlock(128, self.backbone_channels[1], 64)
        self.decoder1 = DecoderBlock(64, self.backbone_channels[0], 32)

        # Final layers
        self.final_conv = ConvBlock(32, 32)

        # Segmentation head (binary lane mask)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

        # Instance embedding head
        self.emb_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, embedding_dim, kernel_size=1)
        )

        self._init_weights()

    def _setup_encoder(self, base_model: nn.Module, backbone: str) -> None:
        """Extract and store encoder blocks from backbone."""
        if backbone == 'efficientnet':
            self.enc0 = nn.Sequential(base_model.features[0])  # stem
            self.enc1 = nn.Sequential(base_model.features[1])  # block0
            self.enc2 = nn.Sequential(base_model.features[2:4])  # blocks1-2
            self.enc3 = nn.Sequential(base_model.features[4:6])  # blocks3-4
            self.enc4 = nn.Sequential(base_model.features[6:9])  # blocks5-7
        else:  # mobilenet
            self.enc0 = nn.Sequential(base_model.features[0])  # first conv
            self.enc1 = nn.Sequential(base_model.features[1])  # block0
            self.enc2 = nn.Sequential(base_model.features[2:4])  # blocks1-2
            self.enc3 = nn.Sequential(base_model.features[4:7])  # blocks3-5
            self.enc4 = nn.Sequential(base_model.features[7:])  # blocks6+

    def _init_weights(self) -> None:
        """Initialize decoder and head weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with:
                'seg': Segmentation logits (B, 1, H, W)
                'emb': Instance embeddings (B, embedding_dim, H, W)
        """
        # Encoder with skip connections
        skip0 = self.enc0(x)  # 1/2
        skip1 = self.enc1(skip0)  # 1/4
        skip2 = self.enc2(skip1)  # 1/8
        skip3 = self.enc3(skip2)  # 1/16
        bottleneck = self.enc4(skip3)  # 1/32

        # Decoder with skip connections
        dec4 = self.decoder4(bottleneck, skip3)  # 1/16
        dec3 = self.decoder3(dec4, skip2)  # 1/8
        dec2 = self.decoder2(dec3, skip1)  # 1/4
        dec1 = self.decoder1(dec2, skip0)  # 1/2

        # Final conv
        x = self.final_conv(dec1)

        # Heads
        seg_logits = self.seg_head(x)
        embeddings = self.emb_head(x)

        return {
            'seg': seg_logits,
            'emb': embeddings
        }

    def print_summary(self) -> None:
        """Print model summary and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"LaneNet Model Summary")
        print(f"{'='*60}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (float32)")
        print(f"{'='*60}\n")

        # Print layer breakdown
        print(f"{'Layer':<30} {'Parameters':<15} {'Trainable':<10}")
        print(f"{'-'*55}")
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:<30} {params:>14,} {str(trainable > 0):<10}")
        print(f"{'-'*55}\n")


def create_lanenet(num_classes: int = 1, backbone: str = 'efficientnet',
                   pretrained: bool = True, embedding_dim: int = 4) -> LaneNet:
    """
    Factory function to create LaneNet model.

    Args:
        num_classes: Number of lane instances
        backbone: Backbone architecture ('efficientnet' or 'mobilenet')
        pretrained: Use pretrained weights
        embedding_dim: Embedding dimension

    Returns:
        LaneNet model instance
    """
    return LaneNet(num_classes=num_classes, backbone=backbone,
                   pretrained=pretrained, embedding_dim=embedding_dim)


if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_lanenet(backbone='efficientnet', pretrained=False)
    model = model.to(device)
    model.print_summary()

    # Forward pass test
    batch_size = 4
    x = torch.randn(batch_size, 3, 384, 640).to(device)
    with torch.no_grad():
        outputs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {outputs['seg'].shape}")
    print(f"Embedding output shape: {outputs['emb'].shape}")
