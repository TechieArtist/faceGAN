import torch
import torch.nn as nn
import torch.nn.functional as F

class build_discriminator(nn.Module):
    def __init__(self, img_shape):
        super(build_discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=img_shape[2], out_channels=64, kernel_size=3, stride=2, padding=1),  # 160x160 -> 80x80
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # 80x80 -> 40x40
            nn.ZeroPad2d((0, 1, 0, 1)),  # 40x40 -> 41x41
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 41x41 -> 21x21
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),  # 21x21 -> 11x11
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Flatten(),  # 512 * 11 * 11
            nn.Linear(512 * 11 * 11, 1),  # 512*11*11 should match the expected input size
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
