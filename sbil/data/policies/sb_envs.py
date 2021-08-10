
import gym
import numpy as np

def identity_policy(obs):
    return obs

def random_policy(obs, observation_space, action_space):
    if isinstance(obs, dict):
        k = next(iter(obs))
        o_size = obs[k].size
        o_shape = obs[k].shape
        o_space_shape = observation_space[k].shape
    else:
        o_size = obs.size
        o_shape = obs.shape
        o_space_shape = observation_space.shape
    action_shape = o_shape[:-len(o_space_shape)] + action_space.shape
    return np.array([action_space.sample() for i in range(o_size // np.prod(o_space_shape))]).reshape(*action_shape)
