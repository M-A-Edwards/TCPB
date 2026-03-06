import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

from models.world_model import WorldModel
from utils.dataset import WorldModelDataset
from configs.config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = WorldModelDataset("../dataset.npz")

model = WorldModel(use_mask=True).to(DEVICE)
model.load_state_dict(torch.load("../sparse.pt", map_location=DEVICE))
model.eval()


obs, action, next_obs = dataset[0]

obs = obs.unsqueeze(0).to(DEVICE)
action = action.unsqueeze(0).to(DEVICE)
next_obs = next_obs.unsqueeze(0).to(DEVICE)


with torch.no_grad():

    z = model.encoder(obs)

importance = []

for i in range(LATENT_DIM):

    z_mod = z.clone()

    z_mod[:, i] = 0

    with torch.no_grad():

        z_next = model.transition(z_mod, action)

        pred = model.decoder(z_next)

    error = torch.mean((pred - next_obs) ** 2).item()

    importance.append(error)


importance = np.array(importance)

plt.bar(range(LATENT_DIM), importance)

plt.title("Latent Dimension Importance")

plt.xlabel("Latent Dimension")
plt.ylabel("Prediction Error when Disabled")

plt.show()