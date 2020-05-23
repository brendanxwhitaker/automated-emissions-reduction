""" RL Environment for automated emissions reduction. """
from typing import Tuple, List
import gym
import numpy as np
from asta import Array

from aer.utils import read_series


class AutomatedEmissionsReductionEnv(gym.Env):
    """
    A gym environment for minimizing CO2 emissions of a refrigerator given MOER values.

    Parameters
    ----------
    source_path : ``str``.
        The path to CSV of MOER values.
    """

    def __init__(self, source_path: str):

        # The observation space is the product of the space of 12d vectors of MOER
        # values and the space of temperatures.
        low = np.array([-1, -1] + ([-1] * 12))
        high = np.array([1, 1] + ([1] * 12))
        self.observation_space = gym.spaces.Box(low, high)

        # Resolution.
        self.res = 12
        self.base_res = 12
        assert self.res % self.base_res == 0
        self.res_scale_factor = self.res // self.base_res

        # The two actions are refrigerator - ``ON`` and ``OFF``.
        self.action_space = gym.spaces.Discrete(2)

        # Load the time series and normalize.
        self.series: Array[float, -1] = read_series(source_path)
        self.series -= 750
        self.series /= 750

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
        ob = np.array([normed_temperature, fridge] + rates)
        return ob

    def reset(self) -> Array[float, 13]:
        """ Resets the environment. """
        self.i = 0
        self.temperature = 33
        self.refrigerating = False

        rates = list(self.series[self.i : self.i + 12])
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
        if self.refrigerating and act == 0:
            self.refrigerating = False
        elif not self.refrigerating and act == 1:
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
        rew = -mwhs * self.series[self.i]
        co2 = mwhs * ((self.series[self.i] * 750) + 750)

        rew *= 10000
        # base_rew = rew
        penalty = 0.0

        if not 33 <= self.temperature <= 43:
            if self.temperature < 33:
                error = 33 - self.temperature
            else:
                error = self.temperature - 43
            penalty = -1 * (1 + error)
            rew += penalty

        # print(f"temp/rew/base_rew/temp_penalty/act: ")
        # print(f"{self.temperature:.3f}/{rew:.3f}/{base_rew:.3f}/{penalty:.3f}/{act}")

        # Increment pointer for emissions rate array, and compute next observation.
        self.i += 1
        rates = list(self.series[self.i : self.i + 12])
        ob = self.get_obs(self.temperature, self.refrigerating, rates)

        done = False
        if self.i + 12 >= len(self.series):
            done = True

        return ob, rew, done, {"co2": co2, "oob": oob}
