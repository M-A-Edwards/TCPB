import torch
import torch.nn as nn

class TransitionModel(nn.Module):

    def __init__(self, latent_dim=64, action_dim=2):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim,128),
            nn.ReLU(),
            nn.Linear(128,latent_dim)
        )

    def forward(self,z,action):

        action_onehot = torch.nn.functional.one_hot(action,2).float()

        x = torch.cat([z,action_onehot],dim=1)

        return self.model(x)

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

class WorldModel(nn.Module):

    def __init__(self, latent_dim=64, use_mask=False):

        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.transition = TransitionModel(latent_dim)

        self.use_mask = use_mask

        if use_mask:
            self.mask = TaskMask(latent_dim)

    def forward(self,obs,action):

        z = self.encoder(obs)

        if self.use_mask:
            z, mask = self.mask(z)
        else:
            mask = None

        z_next = self.transition(z,action)

        pred = self.decoder(z_next)

        return pred, mask