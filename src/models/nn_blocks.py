import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
    )

def _cnn_block(in_ch: int, out_ch: int) -> nn.Module:
    """Conv2d(4x4, stride=2, padding=1) + BatchNorm2d"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ELU(inplace=True),
    )

def _deconv_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ELU(inplace=True),
    )

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.elu(x)