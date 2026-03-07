import torch
from torch.utils.data import Dataset
import numpy as np

class WorldModelDataset(Dataset):

    def __init__(self,path):

        data = np.load(path)

        self.obs = data["obs"]
        self.next_obs = data["next_obs"]
        self.actions = data["actions"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self,idx):

        obs = torch.tensor(self.obs[idx]).unsqueeze(0)
        next_obs = torch.tensor(self.next_obs[idx]).unsqueeze(0)
        action = torch.tensor(self.actions[idx])

        return obs, action, next_obs