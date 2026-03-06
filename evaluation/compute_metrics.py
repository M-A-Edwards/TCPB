import torch
import time
import numpy as np
import sys

sys.path.append("..")

from torch.utils.data import DataLoader

from models.world_model import WorldModel
from utils.dataset import WorldModelDataset
from configs.config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def psnr(mse):

    if mse == 0:
        return 100

    return 20 * np.log10(1.0 / np.sqrt(mse))


def evaluate(model_path, use_mask):

    dataset = WorldModelDataset("../dataset.npz")

    loader = DataLoader(dataset, batch_size=1)

    model = WorldModel(use_mask=use_mask).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()

    total_mse = 0
    total_time = 0

    with torch.no_grad():

        for obs, action, next_obs in loader:

            obs = obs.to(DEVICE)
            action = action.to(DEVICE)
            next_obs = next_obs.to(DEVICE)

            start = time.time()

            pred, mask = model(obs, action)

            end = time.time()

            total_time += end - start

            mse = torch.mean((pred - next_obs) ** 2).item()

            total_mse += mse

    mse = total_mse / len(loader)
    psnr_score = psnr(mse)

    avg_time = total_time / len(loader)

    print("MSE:", mse)
    print("PSNR:", psnr_score)
    print("Inference Time:", avg_time)


if __name__ == "__main__":

    print("Baseline Model")

    evaluate("../baseline.pt", use_mask=False)

    print("\nSparse Model")

    evaluate("../sparse.pt", use_mask=True)