"""
Neural Network Architectures for Lottery Ticket Hypothesis Experiments.

Model A (Conv-4): A 4-convolutional, 2-fully-connected layer CNN closely following
Frankle & Carbin (ICLR 2019) Conv family for CIFAR-10.
Model B (ResNet-9): A compact residual network to study the breakdown of classic
initialization reset (θ0) and demonstrate late rewinding (Frankle et al., 2020).
"""

import torch
import torch.nn as nn
from typing import Dict

class Conv4(nn.Module):
    """
    Model A: 4 Conv Layers (with BatchNorm and ReLU) + 2 Fully Connected Layers.
    Input: (B, 3, 32, 32)
    Output: (B, num_classes)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Conv Block 1: 3 -> 64 -> 64 (spatial: 32x32 -> 16x16)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv Block 2: 64 -> 128 -> 128 (spatial: 16x16 -> 8x8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        # Fully Connected Head: 128 * 8 * 8 = 8192 -> 256 -> num_classes
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Classifier
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def reset_classifier(self, num_classes: int):
        """Replaces the final classification head for transfer learning."""
        self.num_classes = num_classes
        in_features = self.fc2.in_features
        self.fc2 = nn.Linear(in_features, num_classes, bias=True).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


class ResidualBlock(nn.Module):
    """Standard 2-conv residual block with identity shortcut."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNet9(nn.Module):
    """
    Model B: Compact ResNet-9 architecture.
    Useful for studying late rewinding when standard θ0 reset fails on residual networks.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Prep layer: 3 -> 64
        self.prep_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.prep_bn = nn.BatchNorm2d(64)

        # Layer 1: 64 -> 128 + ResBlock
        self.layer1_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(128)
        self.layer1_pool = nn.MaxPool2d(2, 2)
        self.layer1_res = ResidualBlock(128)

        # Layer 2: 128 -> 256
        self.layer2_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_pool = nn.MaxPool2d(2, 2)

        # Layer 3: 256 -> 256 + ResBlock
        self.layer3_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.layer3_bn = nn.BatchNorm2d(256)
        self.layer3_pool = nn.MaxPool2d(2, 2)
        self.layer3_res = ResidualBlock(256)

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.prep_bn(self.prep_conv(x)))
        out = self.layer1_pool(self.relu(self.layer1_bn(self.layer1_conv(out))))
        out = self.layer1_res(out)
        out = self.layer2_pool(self.relu(self.layer2_bn(self.layer2_conv(out))))
        out = self.layer3_pool(self.relu(self.layer3_bn(self.layer3_conv(out))))
        out = self.layer3_res(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def reset_classifier(self, num_classes: int):
        self.num_classes = num_classes
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, num_classes, bias=True).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)


def count_parameters(model: nn.Module, prune_classifier: bool = False) -> Dict[str, Any]:
    """
    Detailed breakdown of model parameter counts:
    - Total parameters
    - Trainable parameters
    - Prunable weights (Conv & Linear weights, excluding classifier if specified)
    - Unpruned parameters (BatchNorm affine weights, biases, classifier if excluded)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    prunable_weight_count = 0
    bn_param_count = 0
    bias_count = 0
    classifier_param_count = 0

    classifier_module = None
    if hasattr(model, 'fc2'):
        classifier_module = model.fc2
    elif hasattr(model, 'fc'):
        classifier_module = model.fc

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module is classifier_module and not prune_classifier:
                classifier_param_count += sum(p.numel() for p in module.parameters())
                continue
            if module.weight is not None:
                prunable_weight_count += module.weight.numel()
            if module.bias is not None:
                bias_count += module.bias.numel()
        elif isinstance(module, nn.BatchNorm2d):
            bn_param_count += sum(p.numel() for p in module.parameters())

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "prunable_weights": prunable_weight_count,
        "bn_parameters": bn_param_count,
        "bias_parameters": bias_count,
        "classifier_parameters": classifier_param_count
    }


def get_model(name: str = "conv4", in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """Factory helper to instantiate models."""
    name_lower = name.lower()
    if name_lower == "conv4":
        return Conv4(in_channels=in_channels, num_classes=num_classes)
    elif name_lower in ["resnet9", "resnet"]:
        return ResNet9(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}. Supported: 'conv4', 'resnet9'")
