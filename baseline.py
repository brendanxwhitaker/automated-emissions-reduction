""" Train the RL agent. """
from aer.env import SimpleEmissionsEnv
from aer.agent import DeterministicAgent

SOURCE_PATH = "data/MOERS.csv"


def main() -> None:
    """ . """
    cutoffs = {}

    for cutoff in range(300, 800, 25):
        env = SimpleEmissionsEnv(SOURCE_PATH)
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
        print("Cutoff:", cutoff, "emissions:", lbs)
        cutoffs[cutoff] = lbs


if __name__ == "__main__":
    main()
