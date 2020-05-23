""" RL Environment for automated emissions reduction. """
from typing import Tuple
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

        # The observation space is the space of 12d vectors of MOER values.
        low = np.zeros((12,))
        high = np.ones((12,)) * 1500
        self.observation_space = gym.spaces.Box(low, high)

        # The two actions are refrigerator - ``ON`` and ``OFF``.
        self.action_space = gym.spaces.Discrete(2)

        # Load the time series.
        series: Array[float, -1] = read_series(source_path)

        self.i = 0

    def reset(self) -> Array[float, 12]:
        """ Resets the environment. """
        raise NotImplementedError

    def step(self, act: int) -> Tuple[Array[float, 12], float, bool, dict]:
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
        raise NotImplementedError
