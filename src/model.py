"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Model
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# IMPROVED BASELINE CNN (Version 2.0)
# ==========================================================

import torch
import torch.nn as nn

from config import NUM_CLASSES


# ==========================================================
# CONVOLUTION BLOCK
# ==========================================================

class ConvBlock(nn.Module):
    """
    Convolution -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


# ==========================================================
# IMPROVED BASELINE CNN
# ==========================================================

class BaselineCNN(nn.Module):
    """
    Improved CNN for CIFAR-10.

    Architecture

    Input (3x32x32)

        Block 1
        Conv32
        Conv32
        MaxPool

        Block 2
        Conv64
        Conv64
        MaxPool

        Block 3
        Conv128
        Conv128
        AdaptiveAvgPool

        FC256
        Dropout

        FC128
        Dropout

        Output10
    """

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # -------------------------------------------------
            # Block 1
            # -------------------------------------------------

            ConvBlock(3, 32),
            ConvBlock(32, 32),

            nn.MaxPool2d(2),

            # -------------------------------------------------
            # Block 2
            # -------------------------------------------------

            ConvBlock(32, 64),
            ConvBlock(64, 64),

            nn.MaxPool2d(2),

            # -------------------------------------------------
            # Block 3
            # -------------------------------------------------

            ConvBlock(64, 128),
            ConvBlock(128, 128),

            nn.AdaptiveAvgPool2d((1, 1))

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(128, NUM_CLASSES)

        )

        self._initialize_weights()

    # ======================================================
    # WEIGHT INITIALIZATION
    # ======================================================

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.constant_(m.bias, 0)

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================================
# GET MODEL
# ==========================================================

def get_model():

    return BaselineCNN()


# ==========================================================
# COUNT PARAMETERS
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# MODEL SUMMARY
# ==========================================================

if __name__ == "__main__":

    model = get_model()

    print(model)

    print(
        f"\nTrainable Parameters : "
        f"{count_parameters(model):,}"
    )

    x = torch.randn(1, 3, 32, 32)

    y = model(x)

    print(f"Output Shape : {y.shape}")