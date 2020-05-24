""" Train the RL agent. """
import json
import random

import torch
import numpy as np
from oxentiel import Oxentiel
from aer.env import AutomatedEmissionsReductionEnv
from aer.trainer import train

SETTINGS_PATH = "settings.json"
SOURCE_PATH = "data/MOERS.csv"

torch.manual_seed(8)
np.random.seed(0)
random.seed(0)


def main() -> None:
    """ Just loads the settings file and calls ``train()``. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    env = AutomatedEmissionsReductionEnv(SOURCE_PATH)
    train(ox, env)


if __name__ == "__main__":
    main()
