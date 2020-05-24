#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Distributed training function for a single agent worker. """
from typing import Dict, Tuple, Any

import torch

from asta import Tensor, dims, typechecked

from aer.bees.rl.storage import RolloutStorage
from aer.bees.rl.algo.algo import Algo

from aer.bees.config import Config
from aer.bees.creation import get_policy
from aer.env import AutomatedEmissionsReductionEnv


# pylint: disable=duplicate-code

STOP_FLAG = 999
SOURCE_PATH = "data/MOERS.csv"

N_ACTS = dims.N_ACTS
FloatTensor = Tensor[float]


@typechecked
def get_masks(
    done: bool, info: Dict[str, Any]
) -> Tuple[FloatTensor[1, 1], FloatTensor[1, 1]]:
    """ Compute masks to insert into ``rollouts``. """
    # If done then clean the history of observations.
    if done:
        masks = torch.FloatTensor([[0.0]])
    else:
        masks = torch.FloatTensor([[1.0]])
    if "bad_transition" in info.keys():
        bad_masks = torch.FloatTensor([[0.0]])
    else:
        bad_masks = torch.FloatTensor([[1.0]])

    return masks, bad_masks


@typechecked
def act(
    iteration: int,
    agent: Algo,
    rollouts: RolloutStorage,
    config: Config,
) -> Tuple[
    FloatTensor[1, 1],
    Tensor[torch.int64, 1, 1],
    FloatTensor[1, 1],
    FloatTensor,
    FloatTensor[1, N_ACTS],
]:
    """ Make a forward pass and send the env action to the leader process. """

    # Rollout tensors have dimension ``0`` size of ``config.num_steps``.
    rollout_index = iteration % config.num_steps
    with torch.no_grad():
        act_returns = agent.actor_critic.act(
            rollouts.obs[rollout_index],
            rollouts.recurrent_hidden_states[rollout_index],
            rollouts.masks[rollout_index],
        )

    return act_returns


def worker_loop(config) -> None:
    """ Training loop for a single agent worker. """
    # Initialize stuff.
    device = torch.device("cpu")
    env = AutomatedEmissionsReductionEnv(SOURCE_PATH)
    agent, rollouts = get_policy(
        config, env.observation_space, env.action_space, device, config.base_kwargs
    )

    i: int = 0

    ob = env.reset()

    # Copy first observations to rollouts, and send to device.
    initial_observation: torch.Tensor = torch.FloatTensor([ob])
    rollouts.obs[0].copy_(initial_observation)
    rollouts.to(device)

    decay: bool = config.use_linear_lr_decay

    rews = []
    oobs = []
    co2s = []

    # Initial forward pass.
    fwds = act(i, agent, rollouts, config)

    for i in range(config.time_steps):
        backward_pass = i % config.num_steps == 0

        # These are all CUDA tensors (on device).
        value: torch.Tensor = fwds[0]
        action: torch.Tensor = fwds[1]
        action_log_prob: torch.Tensor = fwds[2]
        recurrent_hidden_states: torch.Tensor = fwds[3]

        # Get integer action to pass to ``env.step()``.
        env_action: int = int(fwds[1][0])

        # Grab iteration index and env output from leader (no tensors included).
        ob, reward, done, info = env.step(env_action)

        rews.append(reward)
        oobs.append(info["oob"])
        co2s.append(info["co2"])

        decay = config.use_linear_lr_decay and backward_pass

        # If done then remove from environment.
        if done:
            ob = env.reset()
            print(f"Iterations: {i} | Reward: {sum(rews)} | ", end="")
            print(f"OOBs: {sum(oobs)} | CO2 (lbs): {sum(co2s)}", end="")
            print(f"LR: {agent.scheduler.get_lr()}", end="\n")
            rews = []
            oobs = []
            co2s = []

        # Shape correction and casting.
        observation = torch.FloatTensor([ob])
        reward = torch.FloatTensor([reward])
        masks, bad_masks = get_masks(done, info)

        # Add to rollouts.
        rollouts.insert(
            observation,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            reward,
            masks,
            bad_masks,
        )

        # Only when trainer would make an update/backward pass.
        if backward_pass:
            with torch.no_grad():
                next_value = agent.actor_critic.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()
            rollouts.compute_returns(
                next_value,
                config.use_gae,
                config.gamma,
                config.gae_lambda,
                config.use_proper_time_limits,
            )

            # Compute weight updates.
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            rollouts.after_update()

        # Send state back to the leader.
        save_state: bool = i % config.save_interval == 0
        if save_state or i == config.time_steps - 1:
            pass

        # Make a forward pass.
        fwds = act(i, agent, rollouts, config)
