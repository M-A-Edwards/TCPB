import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")

from models.world_model import WorldModel
from utils.dataset import WorldModelDataset
from configs.config import *

dataset = WorldModelDataset("dataset.npz")
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

model = WorldModel(use_mask=True).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=LR)

loss_fn = torch.nn.MSELoss()

for epoch in range(EPOCHS):

    total = 0

    for obs,action,next_obs in loader:

        obs = obs.to(DEVICE)
        next_obs = next_obs.to(DEVICE)
        action = action.to(DEVICE)

        pred,mask = model(obs,action)
        recon_loss = loss_fn(pred,next_obs)
        sparsity = torch.abs(mask).mean()
        loss = recon_loss + SPARSITY_LAMBDA * sparsity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    print("epoch",epoch,"loss",total/len(loader))

torch.save(model.state_dict(),"sparse.pt")