from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
from gym.spaces import Box
import numpy as np
from pathlib import Path
import torch as th
import copy

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, preprocess_obs, get_obs_shape
from stable_baselines3.common.buffers import (
    DictReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    DictRolloutBuffer,
)
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from sbil.utils import get_features_extractor

class AbsorbingState(gym.ObservationWrapper):
    def __init__(self, env):
        super(AbsorbingState, self).__init__(env)
        if isinstance(self.observation_space, Box):
            assert len(self.observation_space.shape) == 1, "For Box spaces, 1D space are only supported"
            self.observation_space = Box(
                low=np.hstack((self.observation_space.low, -1)),
                high=np.hstack((self.observation_space.high, 1)),
                dtype=self.observation_space.dtype,
            )
        else:
            raise NotImplementedError()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observation, reward, done, info = self.env.step(action)
        # indicate absorb: end of episode and no timeout
        self.absorb = done # and not info.get('TimeLimit.truncated', False)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return np.hstack((observation, self.absorb))

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.absorb = False
        return self.observation(observation)



def replay_buffer_with_absorbing(replay_buffer: ReplayBuffer) -> ReplayBuffer:
    """
    Modify observations to indicate absorbing states
    """
    if isinstance(replay_buffer.observation_space, Box) and len((replay_buffer.observation_space.shape))==1:

        o = replay_buffer.observation_space
        assert len(o.shape) == 1, "For Box spaces, 1D spaces are only supported"
        o.high = np.hstack((o.high, 1))
        o.low = np.hstack((o.low, 0))
        o.shape = o.shape[0]+1,
        b = replay_buffer # alias
        absorb = b.dones#*(1-b.timeouts)
        absorb = absorb[:,None]
        if b.optimize_memory_usage:
            absorb = np.roll(absorb, 1, axis=-1)
            b.observations = np.concatenate((b.observations, absorb), axis=-1)
        else:
            b.next_observations = np.concatenate((b.next_observations, absorb), axis=-1)
            b.observations = np.concatenate((b.next_observations, 0), axis=-1) # these are never absorbing
    else:
        raise NotImplementedError()
    return b


def get_demo_buffer(demo_buffer, learner):
    """
    Setup the demo_buffer
    """
    if isinstance(demo_buffer, (str, Path)):
        demo_buffer_ = load_from_pkl(demo_buffer)
    else:
        demo_buffer_ = copy.deepcopy(demo_buffer)

    env = learner.env
    if hasattr(env, "envs"):
        env = env.envs[0]

    if is_wrapped(env, AbsorbingState) and get_flattened_obs_dim(learner.observation_space) == get_flattened_obs_dim(demo_buffer_.observation_space)+1:
        demo_buffer_ = replay_buffer_with_absorbing(demo_buffer_)
    demo_buffer_.device = learner.device
    assert np.isfinite(demo_buffer_.observations).all().item(), "The replay buffer observation contains non-finite values."
    assert np.isfinite(demo_buffer_.actions).all().item(), "The replay buffer actions contains non-finite values."
    check_for_correct_spaces(env, demo_buffer_.observation_space, demo_buffer_.action_space)
    return demo_buffer_

def state_action(state: th.Tensor, action: th.Tensor, learner: BaseAlgorithm, state_only: bool = False):
    """
    Concatenate state and action vectors if not state_only.
    Observations and actions are preprocessed, discrete spaces are converted to one hot
    and they are expected to have a shape (batch_size, *space.shape).
    """
    obs = get_features_extractor(learner).extract_features(state)
    if state_only: return obs
    act = preprocess_obs(action, learner.action_space).view(action.size(0), -1)
    return th.cat((obs, act), dim=-1)

def all_state_action(buffer: RolloutBuffer, learner: BaseAlgorithm, state_only: bool = False):
    """ Equivalent of state_action on the whole RolloutBuffer."""
    o_shape = get_obs_shape(learner.observation_space)
    t = lambda x, shape=[-1]: buffer.to_torch(x).view(buffer.buffer_size*buffer.n_envs, *shape)
    if isinstance(buffer.observations, dict):
        observations = {k: t(v, o_shape[k]) for k, v in buffer.observations.items()} # OrderedDict?
    else:
        observations = t(buffer.observations, o_shape)
    actions = t(buffer.actions)
    return state_action(observations, actions, learner, state_only)
