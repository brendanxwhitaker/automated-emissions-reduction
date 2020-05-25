""" Tests. """
from typing import List
import numpy as np
import hypothesis.strategies as st
from hypothesis import given
from aer.agent import DeterministicAgent

# pylint: disable=unidiomatic-typecheck


@given(
    st.integers(), st.floats(), st.floats(), st.lists(st.integers(), min_size=1),
)
def test_agent(cutoff: int, temp: float, on: float, rates: List[float]) -> None:
    """ Test. """
    ob = np.array([temp, on] + rates)
    agent = DeterministicAgent(cutoff)
    action = agent.act(ob)
    assert type(action) == int
    assert action in (0, 1)
    temp = (temp * 10) + 38
    if temp > 42:
        assert action == 1
    if temp < 34:
        assert action == 0
