from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial

import numpy as np
import gym

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)

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
from sbil.data.policies.sb_envs import identity_policy, random_policy

def scale_action(action: np.ndarray, space) -> np.ndarray:
    if not isinstance(space, gym.spaces.Box): return action
    low, high = space.low, space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0

def generate_demo(env, policy=None, noise=0, buffer_size=100000, device='cpu', optimize_memory_usage=False):
    """
    Return a ReplayBufer filled with demonstrations.
    """
    if isinstance(env, str):
        env = gym.make(env)

    if isinstance(env, VecEnv):
        id = env.get_attr("spec")[0].id
    elif hasattr(env, "envs"):
        id = env.envs[0].unwrapped.spec.id
    else:
        try:
            id = env.unwrapped.spec.id
        except AttributeError:
            id = None

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

    # wrap in VecTransposeImage is needed
    env_ = env
    if not is_vecenv_wrapped(env, VecTransposeImage):
        wrap_with_vectranspose = False
        if isinstance(env.observation_space, gym.spaces.Dict):
            for space in env.observation_space.spaces.values():
                wrap_with_vectranspose = wrap_with_vectranspose or (
                    is_image_space(space) and not is_image_space_channels_first(space)
                )
        else:
            wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                env.observation_space
            )

        if wrap_with_vectranspose:
            if not isinstance(env, VecEnv):
                env_ = DummyVecEnv([lambda: env])
            env_ = VecTransposeImage(env_)


    if isinstance(env_.observation_space, gym.spaces.Dict):
        is_dict = True
        keys = list(env_.observation_space)
    else:
        is_dict = False

    demo_buffer = (DictReplayBuffer if is_dict else ReplayBuffer)(
        buffer_size=buffer_size,
        observation_space=env_.observation_space,
        action_space=env_.action_space,
        device=device,
        optimize_memory_usage=optimize_memory_usage,
    )

    # select policy
    if isinstance(policy, str):
        policy = policy.strip().lower()
        if policy == "identity":
            policy = identity_policy
        elif policy == "random":
            policy = partial(random_policy, observation_space=env_.observation_space, action_space=env_.action_space)
        else:
            raise Exception(f"Unrecognized policy: {policy}. Available policies are identity and random.")
    elif policy is None and id in policies:
        policy = policies[id]
    elif policy is None:
        raise Exception(
            f"The policy is not provided and no policy is available for {id}. "
            f"The following ids have a policy available: {list(policies.keys())}"
        )


    if isinstance(env_, VecEnv):
        num_envs = getattr(env_, "num_envs", 1)
        n = int(np.ceil(buffer_size / num_envs))
        oshape = get_obs_shape(env_.observation_space)

        if isinstance(oshape, dict):
            observations, next_observations = {}, {}
            for k,v in oshape.items():
                observations[k] = np.zeros((n, num_envs) + v)
                next_observations[k] = np.zeros((n, num_envs) + v)
        else:
            observations = np.zeros((n, num_envs) + oshape)
            next_observations = np.zeros((n, num_envs) + oshape)
        actions = np.zeros((n, num_envs, get_action_dim(env.action_space)))
        rewards = np.zeros((n, num_envs, 1))
        dones = np.zeros((n, num_envs, 1))
        infos = np.zeros((n, num_envs, 1), dtype=object)
        obs = env_.reset()

        for i in range(n):
            action = policy(obs)
            next_obs, reward, done, info = env_.step(action)
            if isinstance(obs, dict):
                for k in keys:
                    observations[k][i] = obs[k]
                    next_observations[k][i] = obs[k]
            else:
                observations[i] = obs
                next_observations[i] = next_obs
            actions[i] = action
            rewards[i] = reward
            dones[i] = done
            infos[i] = info
            obs = next_obs

        # indicate last step as truncated
        for info in infos[i]:
            for info_ in info:
                info_['TimeLimit.truncated'] = True

        # scale actions
        actions = scale_action(actions, env_.action_space)

        # flatten envs
        t = lambda x, s=[-1]: x.reshape(n*num_envs, *s)
        if isinstance(observations, dict):
            # reshape observations
            for k in keys:
                observations[k] = t(observations[k], oshape[k])
                next_observations[k] = t(next_observations[k], oshape[k])
            # make it iterable
            obs, next_obs = [{}]*(n*num_envs), [{}]*(n*num_envs)
            for i in range(n*num_envs):
                for k in keys:
                    obs[i][k] = observations[k][i]
                    next_obs[i][k] = next_observations[k][i]
            data = [obs, next_obs,]
        else:
            data = [t(observations, oshape), t(next_observations, oshape)]

        data += list(map(t, (actions, rewards, dones, infos)))

        for obs, next_obs, action, reward, done, info in zip(*data):
            demo_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=info,
            )

    else: # one env, sequential
        while not demo_buffer.full:
            obs = env.reset()
            done = False
            while not done:
                action = policy(obs)
                next_obs, reward, done, info = env_.step(action)
                demo_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=scale_action(action, env_.action_space),
                    reward=reward,
                    done=done,
                    infos=[info],
                )
                obs = next_obs
    return demo_buffer
