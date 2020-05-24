""" A vanilla policy gradient implementation (numpy). """
import time
import gym
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from oxentiel import Oxentiel
from asta import Array, Tensor, shapes, dims

from aer.vpg import ActorCritic, RolloutStorage
from aer.vpg import (
    get_action,
    get_advantages,
    get_rewards_to_go,
    get_policy_loss,
    get_value_loss,
)

SETTINGS_PATH = "settings_vpg.json"

# pylint: disable=invalid-name


def train(ox: Oxentiel, env: gym.Env) -> None:
    """ Trains a policy gradient model with hyperparams from ``ox``. """
    # Set shapes and dimensions for use in type hints.
    dims.RESOLUTION = ox.resolution
    dims.BATCH = ox.batch_size
    dims.ACTS = env.action_space.n
    shapes.OB = env.observation_space.shape

    # Make the policy object.
    ac = ActorCritic(shapes.OB[0], ox.hidden_dim, dims.ACTS)

    # Make optimizers.
    policy_optimizer = Adam(ac.pi.parameters(), lr=ox.lr)
    value_optimizer = Adam(ac.v.parameters(), lr=ox.lr)

    policy_scheduler = OneCycleLR(
        policy_optimizer, ox.lr, ox.lr_cycle_steps, pct_start=ox.pct_start
    )
    value_scheduler = OneCycleLR(
        value_optimizer, ox.lr, ox.lr_cycle_steps, pct_start=ox.pct_start
    )

    # Create a buffer object to store trajectories.
    rollouts = RolloutStorage(ox.batch_size, shapes.OB)

    # Get the initial observation.
    ob: Array[float, shapes.OB]
    ob = env.reset()

    oobs = []
    co2s = []
    mean_co2 = 0
    num_oobs = 0

    t_start = time.time()

    for i in range(ox.iterations):

        # Sample an action from the policy and estimate the value of current state.
        act: Array[int, ()]
        val: Array[float, ()]
        act, val = get_action(ac, ob)

        # Step the environment to get new observation, reward, done status, and info.
        next_ob: Array[float, shapes.OB]
        rew: int
        done: bool
        next_ob, rew, done, info = env.step(int(act))

        # Get co2 lbs.
        co2s.append(info["co2"])
        oobs.append(info["oob"])

        # Add data for a timestep to the buffer.
        rollouts.add(ob, act, val, rew)

        # Don't forget to update the observation.
        ob = next_ob

        # If we reached a terminal state, or we completed a batch.
        if done or rollouts.batch_len == ox.batch_size:

            # Step 1: Compute advantages and critic targets.

            # Get episode length.
            ep_len = rollouts.ep_len
            dims.EP_LEN = ep_len

            # Retrieve values and rewards for the current episode.
            vals: Array[float, ep_len]
            rews: Array[float, ep_len]
            vals, rews = rollouts.get_episode_values_and_rewards()

            mean_rew = np.mean(rews)

            # The last value should be zero if this is the end of an episode.
            last_val: float = 0.0 if done else vals[-1]

            # Compute advantages and rewards-to-go.
            advs: Array[float, ep_len] = get_advantages(ox, rews, vals, last_val)
            rtgs: Array[float, ep_len] = get_rewards_to_go(ox, rews)

            # Record the episode length.
            if done:
                rollouts.lens.append(len(advs))
                rollouts.rets.append(np.sum(rews))

                # Reset the environment.
                ob = env.reset()
                mean_co2 = sum(co2s)
                num_oobs = sum([int(oob) for oob in oobs])
                co2s = []
                oobs = []

            # Step 2: Reset vals and rews in buffer and record computed quantities.
            rollouts.vals[:] = 0
            rollouts.rews[:] = 0

            # Record advantages and rewards-to-go.
            j = rollouts.ep_start
            assert j + ep_len <= ox.batch_size
            rollouts.advs[j : j + ep_len] = advs
            rollouts.rtgs[j : j + ep_len] = rtgs
            rollouts.ep_start = j + ep_len
            rollouts.ep_len = 0

        # If we completed a batch.
        if rollouts.batch_len == ox.batch_size:

            # Get batch data from the buffer.
            obs: Tensor[float, (ox.batch_size, *shapes.OB)]
            acts: Tensor[int, (ox.batch_size)]
            obs, acts, advs, rtgs = rollouts.get_batch()

            # Run a backward pass on the policy (actor).
            policy_optimizer.zero_grad()
            policy_loss = get_policy_loss(ac.pi, obs, acts, advs)
            policy_loss.backward()
            policy_optimizer.step()
            policy_scheduler.step()

            # Run a backward pass on the value function (critic).
            value_optimizer.zero_grad()
            value_loss = get_value_loss(ac.v, obs, rtgs)
            value_loss.backward()
            value_optimizer.step()
            value_scheduler.step()

            # Reset pointers.
            rollouts.batch_len = 0
            rollouts.ep_start = 0

            # Print statistics.
            lr = policy_scheduler.get_lr()
            print(f"Iteration: {i + 1} | ", end="")
            print(f"Time: {time.time() - t_start:.5f} | ", end="")
            print(f"Total co2: {mean_co2:.5f} | ", end="")
            print(f"Num OOBs: {num_oobs:.5f} | ", end="")
            print(f"LR: {lr} | ", end="")
            print(f"Mean reward for current batch: {mean_rew:.5f}")
            t_start = time.time()
            rollouts.rets = []
            rollouts.lens = []
