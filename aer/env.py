""" RL Environment for automated emissions reduction. """
from typing import Tuple, List
import gym
import numpy as np
from asta import Array

from aer.utils import read_series


class TDEmissionsEnv(gym.Env):
    """
    A gym environment for minimizing CO2 emissions of a refrigerator given MOER values.

    Parameters
    ----------
    source_path : ``str``.
        The path to CSV of MOER values.
    """

    def __init__(self, source_path: str):

        # Resolution.
        self.res = 12
        self.base_res = 12
        assert self.res % self.base_res == 0
        self.res_scale_factor = self.res // self.base_res

        # The observation space is the product of the space of 12d vectors of MOER
        # values and the space of temperatures.
        low = np.array([-1, -1] + ([-1] * (self.res - 1)))
        high = np.array([1, 1] + ([1] * (self.res - 1)))
        self.observation_space = gym.spaces.Box(low, high)

        # The two actions are refrigerator - ``ON`` and ``OFF``.
        self.action_space = gym.spaces.Discrete(2)

        # Load the time series and normalize.
        self.series: Array[float, -1] = read_series(source_path)
        self.series -= 750
        self.series /= 750

        self.lb = 34.0
        self.ub = 42.0

        # Repeat each step 5 times to get minute-resolution.
        self.series = self.series.reshape(1, -1)
        self.series = np.tile(self.series, (self.res_scale_factor, 1))
        self.series = np.transpose(self.series)
        self.series = self.series.reshape(-1)

        self.i = 0
        self.power = 200 * 10e-6
        self.temperature = 33.0
        self.refrigerating = False

    def render(self) -> None:
        """ This is just here to make pylint happy. """
        raise NotImplementedError

    @staticmethod
    def get_obs(
        temperature: float, refrigerating: bool, rates: List[float]
    ) -> Array[float, 14]:
        """ Return the observation, with temperature normalized. """
        normed_temperature = (temperature - 38) / 10
        fridge = 1 if refrigerating else -1
        deltas = []
        for i, _ in enumerate(rates[:-1]):
            delta = rates[i + 1] - rates[i]
            deltas.append(delta)
        ob = np.array([normed_temperature, fridge] + deltas)
        return ob

    def reset(self) -> Array[float, 13]:
        """ Resets the environment. """
        self.i = 0
        self.temperature = 33.0
        self.refrigerating = False

        rates = list(self.series[self.i : self.i + self.res])
        ob = self.get_obs(self.temperature, self.refrigerating, rates)
        return ob

    def step(self, act: int) -> Tuple[Array[float, 13], float, bool, dict]:
        """
        Takes an integer action and steps the environment.

        Parameters
        ----------
        act : ``int``.
            The action, either ``0`` for ``OFF`` or ``1`` for ``ON``.

        Returns
        -------
        ob : ``Array[float, 12]``.
            The next set of 12 MOER values in lbs/KWh.
        rew : ``float``.
            The reward for the given action.
        done : ``bool``.
            Whether or not the environment has reached a terminal state.
        info : ``dict``.
            Irrelevant for this problem, used to hold additional information.
        """
        # Turn on/off the refrigerator if necessary.
        if act == 1:
            self.refrigerating = True

        if self.refrigerating:
            self.temperature -= 10 * (1 / self.res)
        else:
            self.temperature += 5 * (1 / self.res)

        self.temperature = max(self.temperature, 28)
        self.temperature = min(self.temperature, 48)

        oob = self.temperature < 33 or self.temperature > 43

        # Compute energy consumption for this interval.
        mwhs = (1 / self.res) * self.power if self.refrigerating else 0

        # Compute CO2 emissions in lbs for this interval.
        # Reward is negative of emissions.
        co2 = mwhs * ((self.series[self.i] * 750) + 750)

        rew = -co2
        rew *= 100
        # base_rew = rew
        penalty = 0.0

        if not self.lb <= self.temperature <= self.ub:
            if self.temperature < self.lb:
                error = self.lb - self.temperature
            else:
                error = self.temperature - self.ub
            penalty = -10 * (1 + error)
            rew += penalty

        # Increment pointer for emissions rate array, and compute next observation.
        self.i += 1
        rates = list(self.series[self.i : self.i + self.res])
        ob = self.get_obs(self.temperature, self.refrigerating, rates)

        done = False
        if self.i + self.res >= len(self.series):
            done = True

        self.refrigerating = False

        return ob, rew, done, {"co2": co2, "oob": oob}


class SimpleEmissionsEnv(TDEmissionsEnv):
    """
    A gym environment for minimizing CO2 emissions of a refrigerator given MOER values.

    Parameters
    ----------
    source_path : ``str``.
        The path to CSV of MOER values.
    """

    def __init__(self, source_path: str):
        super().__init__(source_path)
        low = np.array([-1, -1] + ([-1] * self.res))
        high = np.array([1, 1] + ([1] * self.res))
        self.observation_space = gym.spaces.Box(low, high)

    def render(self) -> None:
        """ This is just here to make pylint happy. """
        raise NotImplementedError

    @staticmethod
    def get_obs(
        temperature: float, refrigerating: bool, rates: List[float]
    ) -> Array[float, 14]:
        """ Return the observation, with temperature normalized. """
        normed_temperature = (temperature - 38) / 10
        fridge = 1 if refrigerating else -1
        ob = np.array([normed_temperature, fridge] + rates)
        return ob
