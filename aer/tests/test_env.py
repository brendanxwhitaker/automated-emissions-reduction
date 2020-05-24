""" Tests. """
import os
import shutil
import tempfile
import datetime
from typing import Callable, Any
import pandas as pd
import hypothesis.strategies as st
import hypothesis.extra.pandas as hpd
from hypothesis import given, assume
from oxentiel import Oxentiel

from aer.env import TDEmissionsEnv, SimpleEmissionsEnv

# pylint: disable=no-value-for-parameter


def padded_strftime(date: datetime.datetime) -> str:
    """ Pad the year. """
    rep = date.strftime("%Y-%m-%d %H:%M:%S")
    year, rest = rep.split("-")[0], rep.split("-")[1:]
    year = "0" * (4 - len(year)) + year
    rep = year + "-" + "-".join(rest)
    return rep


@st.composite
def datasets(draw: Callable[[st.SearchStrategy], Any]) -> pd.DataFrame:
    """ Generates datasets of MOER values. """
    frame = draw(
        hpd.data_frames(
            (hpd.column("timestamp", st.datetimes()), hpd.column("MOER", st.floats()))
        )
    )

    frame.iloc[:, 0] = frame.iloc[:, 0].apply(padded_strftime)
    print(frame)
    return frame


@st.composite
def configs(draw: Callable[[st.SearchStrategy], Any], tempdir: str) -> Oxentiel:
    """ Generates config objects. """
    dataset = draw(datasets())
    source_path = os.path.join(tempdir, "dataset.csv")
    dataset.to_csv(source_path, index=False)
    lr = draw(st.floats(min_value=0, max_value=1))
    lr_cycle_steps = draw(st.integers(min_value=1))
    pct_start = draw(st.floats(min_value=0, max_value=1))
    hidden_dim = draw(st.integers(min_value=1, max_value=8))
    iterations = draw(st.integers(min_value=1, max_value=10000))
    batch_size = draw(st.integers(min_value=1, max_value=100))
    base_resolution = draw(st.integers(min_value=1, max_value=20))
    res_multiplier = draw(st.integers(min_value=1, max_value=10))
    resolution = base_resolution * res_multiplier
    gamma = draw(st.floats(min_value=0, max_value=1))
    lam = draw(st.floats(min_value=0, max_value=1))

    assume(len(dataset) > base_resolution)

    settings = {
        "source_path": source_path,
        "lr": lr,
        "lr_cycle_steps": lr_cycle_steps,
        "pct_start": pct_start,
        "hidden_dim": hidden_dim,
        "iterations": iterations,
        "batch_size": batch_size,
        "resolution": resolution,
        "base_resolution": base_resolution,
        "gamma": gamma,
        "lam": lam,
    }
    ox = Oxentiel(settings)
    return ox


@given(st.data())
def test_td_env(data: st.DataObject) -> None:
    """ Test. """
    tempdir = tempfile.mkdtemp()
    ox = data.draw(configs(tempdir))
    actions = data.draw(st.lists(st.integers(min_value=0, max_value=1)))
    env = TDEmissionsEnv(ox)
    ob = env.reset()
    assert ob.shape == (ox.resolution + 1,)
    for act in actions:
        ob, _rew, done, _info = env.step(act)
        assert ob.shape == (ox.resolution + 1,)
        if done:
            break
    shutil.rmtree(tempdir)


@given(st.data())
def test_simple_env(data: st.DataObject) -> None:
    """ Test. """
    tempdir = tempfile.mkdtemp()
    ox = data.draw(configs(tempdir))
    actions = data.draw(st.lists(st.integers(min_value=0, max_value=1)))
    env = SimpleEmissionsEnv(ox)
    ob = env.reset()
    assert ob.shape == (ox.resolution + 2,)
    for act in actions:
        ob, _rew, done, _info = env.step(act)
        assert ob.shape == (ox.resolution + 2,)
        if done:
            break
    shutil.rmtree(tempdir)
