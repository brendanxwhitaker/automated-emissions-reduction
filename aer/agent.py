""" Deterministic agent for AER. """
from typing import Tuple
import numpy as np
from asta import Array, dims

N = dims.N


class DeterministicAgent:
    """ A baseline agent for automated emissions reduction. """

    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    def act(self, ob: Array[float, N]) -> int:
        """ Given an observation, returns an action. """
        normalized_temperature = float(ob[0])
        normalized_rates = list(ob[2:])

        temperature = (normalized_temperature * 10) + 38
        rates = [int((rate * 750) + 750) for rate in normalized_rates]

        act = 0

        # Deterministic version.
        act = self.deterministic_cutoff(rates[0], self.cutoff)

        if temperature > 42:
            act = 1
        elif temperature < 34:
            act = 0

        return act

    def augment(self, ob: Array[float, N]) -> Tuple[int, bool]:
        """ Given an observation, returns an action and OOB status. """
        normalized_temperature = float(ob[0])
        normalized_rates = list(ob[2:])

        temperature = (normalized_temperature * 10) + 38
        rates = [int((rate * 750) + 750) for rate in normalized_rates]

        act = 0

        # Deterministic version.
        act = self.deterministic_cutoff(rates[0], self.cutoff)

        oob = False
        if temperature > 42:
            act = 1
            oob = True
        elif temperature < 34:
            act = 0
            oob = True

        return act, oob

    @staticmethod
    def deterministic_cutoff(rate: int, cutoff: int) -> int:
        """ Determines action deterministically. """
        act = 0
        if rate < cutoff:
            act = 1
        return act

    @staticmethod
    def probabilistic_cutoff(rate: int, cutoff: int) -> int:
        """ Determines action via Bernoulli trial. """
        act = 0
        if rate < cutoff:
            radius = cutoff
            prob = rate / (2 * radius)
            act = 1
        else:
            radius = 1500 - cutoff
            prob = (1500 - rate) / (2 * radius)
            act = 0

        # Probability we take a suboptimal action.
        outcome = int(np.random.binomial(n=1, p=prob))
        if outcome == 1:
            act = 0 if act == 1 else 1

        return act
