import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256*4*4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,stride=2,padding=1), # 4 -> 8
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,4,stride=2,padding=1),  # 8 -> 16
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,4,stride=2,padding=1),   # 16 -> 32
            nn.ReLU(),

            nn.ConvTranspose2d(32,1,4,stride=2,padding=1),    # 32 -> 64
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1,256,4,4)
        x = self.deconv(x)
        return x