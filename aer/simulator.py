""" Runs a simulation. """
import time
import gym
import numpy as np
import pandas as pd
from asta import Array, shapes
from oxentiel import Oxentiel
from plotplotplot.draw import graph

from aer.agent import Agent

# pylint: disable=invalid-name

SETTINGS_PATH = "settings.json"


def simulate(ox: Oxentiel, env: gym.Env, agent: Agent) -> None:
    """ Trains a policy gradient model with hyperparams from ``ox``. """
    # Types.
    next_ob: Array[float, shapes.OB]
    ob: Array[float, shapes.OB]
    done: bool

    # Get the initial observation.
    ob = env.reset()

    ons = []
    oobs = []
    co2s = []
    temps = []
    moers = []
    total_co2 = 0
    num_oobs = 0

    for i in range(len(env.series)):
        act = agent.act(ob)

        # Step the environment to get new observation, reward, done status, and info.
        next_ob, _, done, info = env.step(act)

        # Record data.
        ons.append(info["on"])
        oobs.append(info["oob"])
        temps.append(info["temp"])
        moers.append(info["moer"])

        # Don't forget to update the observation.
        ob = next_ob

        temp = info["temp"]
        seconds = 3600 * (i + 1) // ox.resolution
        total_co2 += info["co2"]
        co2s.append(total_co2)
        print(f"Iteration: {i + 1} | ", end="")
        print(f"Time elapsed: {seconds}s | ", end="")
        print(f"Total co2: {total_co2:.5f} | ", end="")
        print(f"Temp: {temp:.5f} | ", end="")
        print(f"Num OOBs: {num_oobs:.5f} | ", end="\n")

        if done:
            print("done.")
            break

    # Unnormalize MOERs.
    moers = [rate * 750 + 750 for rate in moers]

    lower = [33.0 for _ in temps]
    upper = [43.0 for _ in temps]

    assert len(temps) == len(moers) == len(ons) == len(co2s) == len(oobs)
    results = [temps, lower, upper, moers, ons, co2s]
    results = [list(row) for row in zip(*results)]
    arr = np.array(results)
    frame = pd.DataFrame(arr)
    frame.columns = ["TEMP", "33F", "43F", "MOER", "ON", "CO2"]
    temp_df = frame.iloc[:, :3]
    moer_df = frame.iloc[:, 3]
    on_df = frame.iloc[:, 4]
    co2_df = frame.iloc[:, 5]
    graph(
        [temp_df, moer_df, on_df, co2_df],
        ["Degrees (F)", "lbs/MWh", "1=ON, 0=OFF", "CO2 in lbs"],
        [3, 1, 1, 1],
        ox.svg_path,
        SETTINGS_PATH,
    )
