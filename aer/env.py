""" RL Environment for automated emissions reduction. """
from typing import Tuple, List
import gym
import numpy as np
from asta import Array, dims, typechecked
from oxentiel import Oxentiel

from aer.utils import read_series

RES = dims.RESOLUTION


@typechecked
class TDEmissionsEnv(gym.Env):
    """
    A gym environment for minimizing CO2 emissions of a refrigerator given MOER values.

    Parameters
    ----------
    ox : ``Oxentiel``.
        A configuration object (corresponds to ``settings.json``).
    """

    def __init__(self, ox: Oxentiel):
        self.res = ox.resolution
        self.base_res = ox.base_resolution

        # Compute the scale factor if we need to upscale resolution of MOER sequence.
        assert self.res % self.base_res == 0
        self.res_scale_factor = self.res // self.base_res

        # See docstring for ``TDEmissionsEnv.get_obs()`` for composition.
        low = np.array([-1] * (self.res + 1))
        high = np.array([1] * (self.res + 1))
        self.observation_space = gym.spaces.Box(low, high)

        # Actions: 0 keeps fridge off, 1 turns fridge on for one timestep.
        self.action_space = gym.spaces.Discrete(2)

        # Load the time series and normalize.
        self.series: Array[float, -1] = read_series(ox.source_path)
        self.series -= 750
        self.series /= 750

        # Set lower and upper bounds for penalty in reward function.
        self.lb = 34.0
        self.ub = 42.0

        # Repeat each step k times to upscale to desired resolution.
        self.series = self.series.reshape(1, -1)
        self.series = np.tile(self.series, (self.res_scale_factor, 1))
        self.series = np.transpose(self.series)
        self.series = self.series.reshape(-1)

        assert len(self.series) >= self.res

        # Initialize the pointer for MOER sequence position and temperature.
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
    ) -> Array[float, RES + 1]:
        """
        Return the observation, with temperature normalized.

        Parameters
        ----------
        temperature : ``float``.
            The temperature in fridge in degrees F.
        refrigerating : ``bool``.
            Whether or not we the fridge is on for this timestep.
        rates : ``List[float]``.
            The MOER values for the next hour.
            Shape: ``(self.res,)``.

        Returns
        -------
        ob : ``Array[float, RES + 1]``.
            Observation of the form:
                [<normed_temp>, <fridge_status>, <differences_of_MOERs>].
            See below for more details.
        """
        normed_temperature = (temperature - 38) / 10
        fridge = 1 if refrigerating else -1

        # Compute the temporal differences between rates.
        deltas = []
        for i, _ in enumerate(rates[:-1]):
            delta = rates[i + 1] - rates[i]
            deltas.append(delta)

        # Now ``len(deltas) == len(rates) - 1``.
        ob = np.array([normed_temperature, fridge] + deltas)
        return ob

    def reset(self) -> Array[float, -1]:
        """ Resets the environment. """
        self.i = 0
        self.temperature = 33.0
        self.refrigerating = False

        rates = list(self.series[self.i : self.i + self.res])
        ob = self.get_obs(self.temperature, self.refrigerating, rates)
        return ob

    def step(self, act: int) -> Tuple[Array[float, -1], float, bool, dict]:
        """
        Takes an integer action and steps the environment.

        Parameters
        ----------
        act : ``int``.
            The action, either ``0`` for ``OFF`` or ``1`` for ``ON``.

        Returns
        -------
        ob : ``Array[float, -1]``.
            See ``get_ob()`` in the relevant environment class.
        rew : ``float``.
            The reward for the given action.
        done : ``bool``.
            Whether or not the environment has reached a terminal state.
        info : ``dict``.
            Used to hold additional information.
        """
        # If action is ``1``, turn on the fridge for one timestep.
        if act == 1:
            self.refrigerating = True

        # Adjust temperature based on whether fridge is ON.
        if self.refrigerating:
            self.temperature -= 10 * (1 / self.res)
        else:
            self.temperature += 5 * (1 / self.res)

        # Keep temperature in a reasonable range so values don't explode.
        self.temperature = max(self.temperature, 28)
        self.temperature = min(self.temperature, 48)

        # Determine if temperature is Out Of Bounds.
        oob: bool = self.temperature < 33 or self.temperature > 43

        # Compute energy consumption for this interval.
        mwhs = (1 / self.res) * self.power if self.refrigerating else 0

        # Compute CO2 emissions in lbs for this interval.
        co2 = mwhs * ((self.series[self.i] * 750) + 750)

        # Reward is scaled negative of emissions.
        rew = -co2
        rew *= 100

        # Compute penalty based on how far temperature is from bounds.
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

        # Determine if we've reached the end of the dataset.
        done = False
        if self.i + self.res >= len(self.series):
            done = True

        # Only keep fridge ON for one step.
        self.refrigerating = False

        return ob, rew, done, {"co2": co2, "oob": oob}


@typechecked
class SimpleEmissionsEnv(TDEmissionsEnv):
    """
    A gym environment for minimizing CO2 emissions of a refrigerator given MOER values.

    Parameters
    ----------
    source_path : ``str``.
        The path to CSV of MOER values.
    """

    def __init__(self, ox: Oxentiel):
        super().__init__(ox)

        # We return the rates instead of differences here, so len(ob) == res + 2.
        low = np.array([-1] * (self.res + 2))
        high = np.array([1] * (self.res + 2))
        self.observation_space = gym.spaces.Box(low, high)

    def render(self) -> None:
        """ This is just here to make pylint happy. """
        raise NotImplementedError

    @staticmethod
    def get_obs(
        temperature: float, refrigerating: bool, rates: List[float]
    ) -> Array[float, RES + 2]:
        """
        Return the observation, with temperature normalized.

        Parameters
        ----------
        temperature : ``float``.
            The temperature in fridge in degrees F.
        refrigerating : ``bool``.
            Whether or not we the fridge is on for this timestep.
        rates : ``List[float]``.
            The MOER values for the next hour.
            Shape: ``(self.res,)``.

        Returns
        -------
        ob : ``Array[float, RES + 2]``.
            Observation of the form:
                [<normed_temp>, <fridge_status>, <MOERs>].
        """
        normed_temperature = (temperature - 38) / 10
        fridge = 1 if refrigerating else -1
        ob = np.array([normed_temperature, fridge] + rates)
        return ob
