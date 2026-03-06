import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder
from models.transition_model import TransitionModel
from models.task_mask import TaskMask

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