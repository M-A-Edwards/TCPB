import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, latent_dim=64):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128*6*6)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2),
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2),
            nn.Sigmoid()
        )

    def forward(self,z):

        x = self.fc(z)
        x = x.view(-1,128,6,6)
        x = self.deconv(x)

        return x