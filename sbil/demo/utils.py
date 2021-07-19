from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
from gym.spaces import Box
import numpy as np
from pathlib import Path
import torch as th
import copy

from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
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

class AbsorbingState(gym.ObservationWrapper):
    def __init__(self, env):
        super(AbsorbingState, self).__init__(env)
        if isinstance(self.observation_space, Box):
            assert len(self.observation_space.shape) == 1, "For Box spaces, 1D space are only supported"
            self.observation_space = Box(
                low=np.hstack((self.observation_space.low,-1)),
                high=np.hstack((self.observation_space.high,1)),
                dtype=self.observation_space.dtype,
            )
        else:
            raise NotImplementedError()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observation, reward, done, info = self.env.step(action)
        # indicate absorb: end of episode and no timeout
        self.absorb = done and not info.get('TimeLimit.truncated', False)
        return self.observation(observation), reward, done, info
    
    def observation(self, observation):
        return np.hstack((observation, self.absorb))

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.absorb = False
        return self.observation(observation)



def replay_buffer_with_absorbing(replay_buffer: ReplayBuffer) -> ReplayBuffer:
    if isinstance(replay_buffer.observation_space, Box) and len((replay_buffer.observation_space.shape))==1:
        
        o = replay_buffer.observation_space
        assert len(o.shape) == 1, "For Box spaces, 1D spaces are only supported"
        o.high = np.hstack((o.high, 1))
        o.low = np.hstack((o.low, 0))
        o.shape = o.shape[0]+1,
        b = replay_buffer # alias
        absorb = b.dones*(1-b.timeouts)
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
    """
    observation_space = Box(
            low=np.hstack((replay_buffer.observation_space.low,-1)),
            high=np.hstack((replay_buffer.observation_space.high,1)),
            dtype=replay_buffer.observation_space.dtype,
        )
    new_replay_buffer = ReplayBuffer(
        buffer_size=replay_buffer.buffer_size,
        observation_space=observation_space,
        action_space=replay_buffer.action_space,
        optimize_memory_usage=replay_buffer.optimize_memory_usage)
    for i in range(replay_buffer.size()-1):
        new_replay_buffer.add(
            obs=np.hstack((replay_buffer.observations[i], replay_buffer.dones[i][None])),
            next_obs=np.hstack((
                replay_buffer.observations[i+1]  if replay_buffer.optimize_memory_usage
                else replay_buffer.next_observations[i], replay_buffer.dones[i+1][None]
            )),
            action=replay_buffer.actions[i],
            reward=replay_buffer.rewards[i],
            done=replay_buffer.dones[i],
            infos=[{}],
        )
    
    return new_replay_buffer
    """

def get_demo_buffer(demo_buffer, learner):
    if isinstance(demo_buffer, (str, Path)):
        demo_buffer_ = load_from_pkl(demo_buffer)
    else:
        demo_buffer_ = copy.deepcopy(demo_buffer)
    if get_flattened_obs_dim(learner.observation_space) == get_flattened_obs_dim(demo_buffer_.observation_space)+1:
        demo_buffer_ = replay_buffer_with_absorbing(demo_buffer_)
    demo_buffer_.device = learner.device
    return demo_buffer_

def state_action(state, action, state_only=False, numpy=True):
    t = (lambda x: np.array(x)) if numpy else lambda x: th.as_tensor(x, dtype=th.float32)
    sa = (t(state), () if state_only else t(action))
    return np.concatenate(sa, axis=-1) if numpy else th.cat(sa, dim=-1)

def all_state_action(buffer, extract_features, state_only):
    view = buffer.buffer_size*buffer.n_envs, -1
    if isinstance(buffer.observations, dict):
        observations = {key: buffer.to_torch(obs).view(*view) for (key, obs) in buffer.observations.items()} # OrderedDict?
    else:
        observations = buffer.to_torch(buffer.observations).view(*view)
    observations = extract_features(observations)
    actions = buffer.to_torch(buffer.actions).view(*view)
    return state_action(observations, actions, state_only, numpy=False)
