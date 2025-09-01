import torch
import torch.nn as nn
from .model import SpatialTransformer
from . import losses


class WeightNet(nn.Module):
    """Lightweight network predicting spatially varying weights."""

    def __init__(self, in_channels=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 3, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x):
        w = self.net(x)
        return torch.chunk(w, 3, dim=1)


class BoidsRegister(nn.Module):
    """VoxelMorph variant using Boids-inspired iterative updates."""

    def __init__(self, vol_size, in_channels=2, steps=5, step_size=1.0, weight_net=None):
        super().__init__()
        self.transformer = SpatialTransformer(vol_size)
        self.steps = steps
        self.step_size = step_size
        self.weight_net = weight_net or WeightNet(in_channels)

    def forward(self, moving, fixed):
        batch = moving.shape[0]
        device = moving.device
        v = torch.zeros(batch, 3, *moving.shape[2:], device=device, requires_grad=True)
        features = torch.cat([moving, fixed], dim=1)
        alpha, beta, gamma = self.weight_net(features)
        for _ in range(self.steps):
            v.requires_grad_(True)
            warped = self.transformer(moving, v)
            sim_loss = losses.ncc_loss(warped, fixed)
            F_data = -torch.autograd.grad(sim_loss, v, create_graph=True)[0]
            align_loss = losses.gradient_loss(v)
            F_align = -torch.autograd.grad(align_loss, v, create_graph=True)[0]
            jac_loss = losses.NJ_loss(v)
            F_sep = -torch.autograd.grad(jac_loss, v, create_graph=True)[0]
            update = alpha * F_data + beta * F_align + gamma * F_sep
            v = v + self.step_size * update
        warped = self.transformer(moving, v)
        return warped, v
