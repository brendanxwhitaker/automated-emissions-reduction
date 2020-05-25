""" Runs a simulation. """
import json
import random

import torch
import numpy as np

from oxentiel import Oxentiel

from aer.env import TDEmissionsEnv, SimpleEmissionsEnv
from aer.agent import DeterministicAgent, VPGAgent, Agent
from aer.simulator import simulate

SETTINGS_PATH = "settings.json"

torch.manual_seed(8)
np.random.seed(0)
random.seed(0)


def main() -> None:
    """ Just loads the settings file and calls ``train()``. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)

    # Get the specified agent/env.
    agent: Agent
    if ox.agent == "vpg":
        env = TDEmissionsEnv(ox)
        agent = VPGAgent(ox)
    elif ox.agent == "deterministic":
        env = SimpleEmissionsEnv(ox)
        agent = DeterministicAgent(ox.cutoff)
    else:
        raise ValueError("Invalid agent choice. Should be 'vpg' or 'deterministic'.")

    # Run the simulation.
    simulate(ox, env, agent)


if __name__ == "__main__":
    main()
