""" Deterministic agent for AER. """
from abc import abstractmethod
import torch
from asta import Array, dims, typechecked
from oxentiel import Oxentiel
from aer.vpg import get_action

# Asta dimension for the observation resolution.
RES = dims.RESOLUTION

# pylint: disable=invalid-name, too-few-public-methods


class Agent:
    """ Agent ABC. """

    @abstractmethod
    def act(self, ob: Array[float, -1]) -> int:
        """ Given an observation, returns an action. """
        raise NotImplementedError


@typechecked
class DeterministicAgent(Agent):
    """ A baseline agent for automated emissions reduction. """

    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    def act(self, ob: Array[float, RES + 2]) -> int:
        """ Given an observation, returns an action. """
        normalized_temperature = float(ob[0])
        normalized_rates = list(ob[2:])

        # Unnormalize the temperature.
        temperature = (normalized_temperature * 10) + 38

        # Unnormalize the MOER rates.
        rates = [int((rate * 750) + 750) for rate in normalized_rates]

        # The default action is to do nothing (fridge OFF).
        act = 0

        # Turn the fridge on for one timestep if rate is below cutoff.
        act = self.deterministic_cutoff(rates[0], self.cutoff)

        # Override chosen action if we are in danger zone.
        if temperature > 42:
            act = 1
        elif temperature < 34:
            act = 0

        return act

    @staticmethod
    def deterministic_cutoff(rate: int, cutoff: int) -> int:
        """ Determines action deterministically. """
        act = 0
        if rate < cutoff:
            act = 1
        return act


class VPGAgent(Agent):
    """ A wrapper around an ``ActorCritic`` module. """

    def __init__(self, ox: Oxentiel):
        with open(ox.load_path, "rb") as model_file:
            self.ac = torch.load(model_file)

    def act(self, ob: Array[float, RES + 1]) -> int:
        """ Given an observation, returns an action. """
        act, _val = get_action(self.ac, ob)
        return int(act)
