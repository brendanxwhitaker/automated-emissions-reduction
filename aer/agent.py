""" Deterministic agent for AER. """
from asta import Array, dims, typechecked

N = dims.N


@typechecked
class DeterministicAgent:
    """ A baseline agent for automated emissions reduction. """

    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    def act(self, ob: Array[float, 14]) -> int:
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

    @staticmethod
    def deterministic_cutoff(rate: int, cutoff: int) -> int:
        """ Determines action deterministically. """
        act = 0
        if rate < cutoff:
            act = 1
        return act
