import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, latent_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,32,4,2),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,128,4,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*6*6, latent_dim)
        )

    def forward(self,x):
        return self.net(x)