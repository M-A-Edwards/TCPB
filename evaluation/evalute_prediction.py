import torch
import sys
import matplotlib.pyplot as plt

sys.path.append("..")

from models.world_model import WorldModel
from utils.dataset import WorldModelDataset
from configs.config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path, use_mask):

    model = WorldModel(use_mask=use_mask).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    return model


def visualize_prediction(model, dataset, idx=0):

    obs, action, next_obs = dataset[idx]

    obs = obs.unsqueeze(0).to(DEVICE)
    action = action.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred, mask = model(obs, action)

    pred = pred.cpu().squeeze().numpy()
    gt = next_obs.squeeze().numpy()
    inp = obs.cpu().squeeze().numpy()

    fig, axs = plt.subplots(1,3)

    axs[0].set_title("Input")
    axs[0].imshow(inp, cmap="gray")

    axs[1].set_title("Ground Truth")
    axs[1].imshow(gt, cmap="gray")

    axs[2].set_title("Prediction")
    axs[2].imshow(pred, cmap="gray")

    plt.show()


if __name__ == "__main__":

    dataset = WorldModelDataset("../dataset.npz")

    model = load_model("../sparse.pt", use_mask=True)

    visualize_prediction(model, dataset, idx=10)