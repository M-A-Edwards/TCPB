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