import torch
import torch.nn as nn

class build_generator(nn.Module):
    def __init__(self, seed_size, channels):
        super(build_generator, self).__init__()

        self.model = nn.Sequential(
            # Dense layer
            nn.Linear(seed_size, 5 * 5 * 256),
            nn.ReLU(inplace=True),

            # Reshape to 5x5x256
            nn.Unflatten(1, (256, 5, 5)),

            # Upsample to 10x10
            nn.Upsample(scale_factor=2, mode='nearest'),  # Use nearest neighbor upsampling
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.8),
            nn.ReLU(inplace=True),

            # Upsample to 20x20
            nn.Upsample(scale_factor=2, mode='nearest'),  # Use nearest neighbor upsampling
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.ReLU(inplace=True),

            # Upsample to 40x40
            nn.Upsample(scale_factor=2, mode='nearest'),  # Use nearest neighbor upsampling
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(inplace=True),

            # Upsample to 80x80
            nn.Upsample(scale_factor=2, mode='nearest'),  # Use nearest neighbor upsampling
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(inplace=True),

            # Upsample to 160x160
            nn.Upsample(scale_factor=2, mode='nearest'),  # Use nearest neighbor upsampling
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(inplace=True),

            # Final layer
            nn.Conv2d(128, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


