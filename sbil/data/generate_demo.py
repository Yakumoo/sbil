from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.vec_env import VecEnv

from sbil.data.policies.gym_solutions import (
    cartpole,
    mountain_car,
    mountain_car_continuous,
    pendulum,
    acrobot,
    lunar_lander,
    lunar_lander_continuous,
    bipedal_walker,
)

def generate_demo(env, policy=None, noise=0, buffer_size=10000, device='cpu', optimize_memory_usage=False):
    """
    Return a ReplayBufer filled with demonstrations.
    """
    if isinstance(env, VecEnv):
        id = env.get_attr("spec")[0].id
    elif hasattr(env, "envs"):
        id = env.envs[0].unwrapped.spec.id
    else:
        id = env.unwrapped.spec.id

    policies = {
        'Acrobot-v1': acrobot,
        'BipedalWalker-v3': bipedal_walker,
        'CartPole-v0': cartpole,
        'LunarLander-v2': lunar_lander,
        'LunarLanderContinuous-v2': lunar_lander_continuous,
        'MountainCar-v0': mountain_car,
        'MountainCarContinuous-v0': mountain_car_continuous,
        'Pendulum-v0': pendulum,

    }

    demo_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        optimize_memory_usage=optimize_memory_usage,
    )

    if policy is None and id in policies:
        policy = policies[id]
    elif policy is None:
        raise Exception(
            f"The policy is not provided and no policy is available for {id}. "
            f"The following ids have a policy available: {list(policies.keys())}"
        )


    if isinstance(env, VecEnv):
        num_envs = getattr(env, "num_envs", 1)
        n = int(np.ceil(buffer_size / num_envs))
        oshape = get_obs_shape(env.observation_space)
        observations = np.zeros((n, num_envs) + oshape)
        next_observations = np.zeros((n, num_envs) + oshape)
        actions = np.zeros((n, num_envs, get_action_dim(env.action_space)))
        rewards = np.zeros((n, num_envs, 1))
        dones = np.zeros((n, num_envs, 1))
        infos = np.zeros((n, num_envs, 1), dtype=object)
        obs = env.reset()
        for i in range(n):
            action = policy(obs)
            next_obs, reward, done, info = env.step(action)
            observations[i] = obs
            next_observations[i] = next_obs
            actions[i] = action
            rewards[i] = reward
            dones[i] = done
            infos[i] = info
            obs = next_obs
        for info in infos[i]: # indicate last step as truncated
            for info_ in info:
                info_['TimeLimit.truncated'] = True
        # flatten envs
        data = (observations, next_observations, actions, rewards, dones, infos)
        data = list(map(lambda x: x.reshape(n*num_envs, -1), data))
        #observations, next_observations, actions, rewards, dones, infos = data
        for obs, next_obs, action, reward, done, info in zip(*data):
            demo_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=[info[0]],
            )
    else: # one nev, sequential
        while not demo_buffer.full:
            obs = env.reset()
            done = False
            while not done:
                action = policy(obs)
                next_obs, reward, done, info = env.step(action)
                demo_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos=[info],
                )
                obs = next_obs
    return demo_buffer
