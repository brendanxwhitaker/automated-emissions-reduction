""" Train the RL agent. """
import json
from oxentiel import Oxentiel
from aer.env import AutomatedEmissionsReductionEnv, SimplifiedAEREnv
from aer.augmented_trainer import train

SETTINGS_PATH = "settings_vpg.json"
SOURCE_PATH = "data/MOERS.csv"


def main() -> None:
    """ Just loads the settings file and calls ``train()``. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    env = SimplifiedAEREnv(SOURCE_PATH)
    train(ox, env)


if __name__ == "__main__":
    main()
