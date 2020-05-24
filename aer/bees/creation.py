#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Functions for agent instantiation. """
from typing import Dict, Tuple, Any

import gym
import torch

from aer.bees.rl.algo import Algo, A2C_ACKTR
from aer.bees.ppo import PPO
from aer.bees.rl.model import Policy
from aer.bees.rl.storage import RolloutStorage

from aer.bees.config import Config


def get_policy(
    config: Config,
    obs_space: gym.Space,
    act_space: gym.Space,
    device: torch.device,
    base_kwargs: Dict[str, Any],
) -> Tuple[Algo, RolloutStorage]:
    """
    Spins up a new agent/policy.

    Parameters
    ----------
    config : ``Config``.
        Config object parsed from settings file.
    obs_space : ``gym.Space``.
        Observation space from the environment.
    act_space : ``gym.Space``.
        Action space from the environment.
    device : ``torch.device``.
        The GPU/TPU/CPU.
    base_kwargs : ``Dict[str, Any]
        Arguments to ``self.base`` (either CNNBase or MLPBase).

    Returns
    -------
    agent : ``Algo``.
        Agent object from a2c-ppo-acktr.
    rollouts : ``RolloutStorage``.
        The rollout object.
    """

    actor_critic = Policy(obs_space.shape, act_space, base_kwargs=base_kwargs)
    actor_critic.to(device)
    agent: Algo

    if config.algo == "a2c":
        agent = A2C_ACKTR(
            actor_critic,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            alpha=config.alpha,
            max_grad_norm=config.max_grad_norm,
        )
    elif config.algo == "ppo":
        agent = PPO(actor_critic, config,)
    elif config.algo == "acktr":
        agent = A2C_ACKTR(
            actor_critic, config.value_loss_coef, config.entropy_coef, acktr=True
        )

    rollouts = RolloutStorage(
        config.num_steps,
        config.num_processes,
        obs_space.shape,
        act_space,
        actor_critic.recurrent_hidden_state_size,
    )
    return agent, rollouts
