import gymnasium as gym
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import preprocess_frame
from configs.config import DATASET_SIZE

env = gym.make("CarRacing-v2", render_mode="rgb_array")
obs_list = []
next_list = []
actions = []

obs, _ = env.reset()

for _ in tqdm(range(DATASET_SIZE)):

    frame = env.render()
    frame = preprocess_frame(frame)

    action = env.action_space.sample()

    next_obs, _, done, _, _ = env.step(action)

    next_frame = env.render()
    next_frame = preprocess_frame(next_frame)

    obs_list.append(frame)
    next_list.append(next_frame)
    actions.append(action)

    if done:
        env.reset()

np.savez(
    "dataset.npz",
    obs=np.array(obs_list),
    next_obs=np.array(next_list),
    actions=np.array(actions, dtype=np.float32),
)