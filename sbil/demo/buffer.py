
# SQIL
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from sbil.utils import set_method, set_restore
from sbil.demo.utils import get_demo_buffer

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import torch as th
import numpy as np
import gym


def sample(self, batch_size, env, *args, super_, demo_buffer, policy_reward=None, demo_rate=0.5, **kwargs) -> Union[DictReplayBufferSamples, ReplayBufferSamples]:
    demo_size = int(batch_size*demo_rate)
    demo_sample = demo_buffer.sample(batch_size=demo_size, env=env)
    replay_data = super_(batch_size=batch_size-demo_size, env=env, *args, **kwargs)

    if policy_reward is not None:
        replay_data.rewards[:] = policy_reward

    data = (
        (replay_data.actions, demo_sample.actions),
        (replay_data.next_observations, demo_sample.next_observations),
        (replay_data.dones, demo_sample.dones),
        (replay_data.rewards, demo_sample.rewards),
    )
    if isinstance(replay_data.observations, dict):
        observations = {key:th.cat((obs, demo_sample.observations[key]), dim=0) for key, obs in replay_data.observations.items()}
        return DictReplayBufferSamples(observations, *tuple(map(th.cat, data)))
    else:
        data = ((replay_data.observations, demo_sample.observations),) + data
        return ReplayBufferSamples(*tuple(map(th.cat, data)))

def double_buffer( # SQIL
    learner: OffPolicyAlgorithm,
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    demo_reward = None,
    policy_reward = None,
    demo_rate: float = 0.5
) -> OffPolicyAlgorithm:
    """
    Double buffer decorator
    Half of sample is from the replay_buffer (dynamic)
    Other half if from the demo_buffer (fixed) during sampling

    :param learner: stable baselines learner object
    :param demo_buffer: demonstration replay buffer
    :demo_reward: set the reward of the demonstration to a constant
    :policy_reward: set the reward of the policy to a constant
        if demo_reward=1 and policy_reward=0, this is SQIL with λ=1.
        https://arxiv.org/abs/1905.11108
        Default is unchanged (rewards of the environment).
    :return leaner: decorated learner
    """
    assert 0 < demo_rate < 1, "demo_rate must be in [0, 1]."
    demo_buffer = get_demo_buffer(demo_buffer, learner)
    if demo_reward is not None:
        demo_buffer.rewards[:] = demo_reward
    set_restore(learner.replay_buffer)

    set_method(
        learner.replay_buffer,
        old="sample",
        new=sample,
        demo_buffer=demo_buffer,
        policy_reward=policy_reward,
        demo_rate=demo_rate,
    )
    return learner
