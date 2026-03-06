import torch
import torch.nn as nn

class TaskMask(nn.Module):

    def __init__(self, latent_dim=64):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z):

        mask_logits = self.net(z)

        mask = torch.sigmoid(mask_logits)

        return z * mask, mask