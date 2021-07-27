
# Maximum Mean Discrepancy
# GMMIL

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from sbil.utils import set_method, set_restore
from sbil.demo.utils import get_demo_buffer, state_action, all_state_action

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import torch as th
import numpy as np
import gym



def compute_returns_and_advantage(self, super_, demo_buffer, learner, state_only: bool = False, max_size=-1, σ2=None, *args, **kwargs) -> None:
    if max_size > 1: # subsample
        #view = self.buffer_size*self.n_envs, -1
        demo_sample = demo_buffer.sample(max_size)
        demo_sa = state_action(demo_sample.observations, demo_sample.actions, learner, state_only, numpy=False)
        σ2 = np.median(np.square(np.linalg.norm(demo_sa-demo_sa[:,None], axis=-1))).item()
    else: # use all, high memory consumption
        σ2 = σ2 # σ2 is precomputed
        demo_sa = all_state_action(demo_buffer, learner, state_only)

    sa = all_state_action(self, learner, state_only)
    d1 = np.square(np.linalg.norm(sa-demo_sa[:,None], axis=-1)) # distance matrix
    σ1 = np.median(d1).item()
    σ = np.reshape([σ1, σ2], (2,1,1))
    d2 = np.square(np.linalg.norm(sa-sa[:,None], axis=-1))
    # axis0 = 2 (σ1, σ2), axis1 = max_size, axis2 = buffer_size*n_envs
    rewards = np.mean(np.exp(- d1/σ), axis=(0,1)) - np.mean(np.exp(-d2/σ), axis=(0,1))
    self.rewards[:] = rewards.reshape(self.buffer_size, self.n_envs) # back to original shape
    super_(*args, **kwargs)

def gmmil(
    learner: OnPolicyAlgorithm,
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    state_only: bool = False,
    max_size: int = -1
) -> OnPolicyAlgorithm:
    """
    GMMIL decorator: https://www-users.cs.umn.edu/~hspark/mmd.pdf
    The reward is modified in learner.rollout_buffer.compute_returns_and_advantage

    :param learner: stable baselines learner object
    :param demo_buffer: demonstration replay buffer
    :param state_only: default is the concatenation of the state-action pair
    :param max_size: Subsample a batch of size max_size from demo_buffer to save memory
        Default uses the whole buffer, high memory consumption due to pairwise distance
    :return leaner: decorated learner
    """

    demo_buffer = get_demo_buffer(demo_buffer, learner)
    set_restore(learner.rollout_buffer)

    extract_features = learner.policy.extract_features

    # median heuristic, σ2 is precomputed as it doesn't change if max_size<0
    if max_size < 0:
        demo_sa = all_state_action(demo_buffer, learner, state_only=state_only)
        σ2 = np.median(np.square(np.linalg.norm(demo_sa-demo_sa[:,None], axis=-1))).item()
    else:
        σ2 = None

    set_method(
        learner.rollout_buffer,
        old="compute_returns_and_advantage",
        new=compute_returns_and_advantage,
        demo_buffer=demo_buffer,
        σ2=σ2,
        max_size=max_size,
        learner=learner,
    )
    return learner
