import torch
from torch.utils.data import Dataset
import numpy as np

class WorldModelDataset(Dataset):
    def __init__(self, file):

        data = np.load(file)

        self.obs = data["obs"]
        self.next_obs = data["next_obs"]
        self.actions = data["actions"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):

        o = torch.tensor(self.obs[idx]).unsqueeze(0)
        n = torch.tensor(self.next_obs[idx]).unsqueeze(0)
        a = torch.tensor(self.actions[idx]).long()

        return o, a, n