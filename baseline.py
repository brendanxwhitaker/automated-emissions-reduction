""" Train the RL agent. """
import json
from asta import dims
from oxentiel import Oxentiel
from aer.env import SimpleEmissionsEnv
from aer.agent import DeterministicAgent

SETTINGS_PATH = "settings.json"


def main() -> None:
    """ . """
    # Read in the settings file.
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)

    # Set typecheck variables.
    dims.RESOLUTION = ox.resolution

    # Map cutoff values to the resulting total emissions.
    cutoffs = {}

    # Search for the optimal cutoff MOER value.
    for cutoff in range(500, 700, 10):
        env = SimpleEmissionsEnv(ox)
        agent = DeterministicAgent(cutoff=cutoff)

        emissions = []
        ob = env.reset()
        for _ in range(200000):
            act = agent.act(ob)
            ob, _, done, info = env.step(act)

            co2 = info["co2"]
            emissions.append(co2)

            if done:
                break

        lbs = sum(emissions)
        print("|||||||||||||||||||||||| Cutoff:", cutoff, "emissions:", lbs)
        cutoffs[cutoff] = lbs


if __name__ == "__main__":
    main()
