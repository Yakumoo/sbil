from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import io
import os
from argparse import ArgumentParser
import yaml
import importlib
from types import MethodType
from functools import partial
import inspect
from zipfile import ZipFile
from operator import attrgetter

import gym
from gym.spaces import Box
import pandas as pd
import numpy as np
import torch as th
import stable_baselines3 as sb


import stable_baselines3 as sb
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.policies import BaseModel, get_policy_from_name
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution, SquashedDiagGaussianDistribution, CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv, VecNormalize
import sbil


def safe_eval(s:str):
    global_symbols = { # whitelist
        # available builtins
        '__builtins__': {k: __builtins__[k] for k in
            ['list', 'dict', 'map', 'len', 'str', 'float', 'int', 'True', 'False', 'min', 'max', 'round']
        },
        # available modules
        'np':np,
        'th':th,
        'sb':sb,
        'pd':pd,
        'gym':gym,
    }
    return exec(s, global_symbols)

class TimeLimitAware(gym.ObservationWrapper):
    """ Copy paste from https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py """
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        
        if isinstance(self.observation_space, Box):
            assert len(self.observation_space.shape) == 1, "For Box spaces, 1D spaces are only supported"
            self.observation_space = Box(
                low=np.hstack((self.observation_space.low,-1)),
                high=np.hstack((self.observation_space.high,1)),
                dtype=self.observation_space.dtype,
            )
        else:
            raise NotImplementedError()

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return self.observation(observation), reward, done, info
    
    def observation(self, observation):
        # -1: beginning, 1: end
        scaled_remaining = (self._elapsed_steps/self._max_episode_steps-1)*2+1
        return np.hstack((observation, scaled_remaining))

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class MLP(th.nn.Module):
    def __init__(self, input_dim, output_dim, net_arch=[32,32], output_transform=None, spectral_norm=False, optimizer={'class':th.optim.Adam}):
        super(MLP, self).__init__()

        #self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.net_arch = net_arch
        self.output_transform = output_transform
        distributions = {
            "DiagGaussianDistribution": DiagGaussianDistribution,
            "SquashedDiagGaussianDistribution": SquashedDiagGaussianDistribution,
            "CategoricalDistribution": CategoricalDistribution,
            "MultiCategoricalDistribution": MultiCategoricalDistribution,
            "BernoulliDistribution": BernoulliDistribution,
            "StateDependentNoiseDistribution": StateDependentNoiseDistribution,
        }
        #self.action_dim = get_action_dim(self.action_space)
        #output_dim = self.action_dim
        if output_transform in distributions:
            self.distribution = distributions[output_transform](self.action_dim)
            if output_transform in {"DiagGaussianDistribution", "SquashedDiagGaussianDistribution"}: output_dim = self.action_dim*2
        else:
            self.distribution = None

        layers = create_mlp(input_dim=input_dim, output_dim=output_dim, net_arch=net_arch, activation_fn=th.nn.LeakyReLU)
        if spectral_norm:
            i = 0
            for layer in layers:
                if isinstance(layer, th.nn.Linear) and i < len(net_arch):
                    i += 1
                    layer = th.nn.utils.spectral_norm(layer) # use parametrizations
        if output_transform is not None and output_transform not in distributions:
            layers += [getattr(th.nn, output_transform)() if isinstance(output_transform, str) else output_transform()]
        self.mlp = th.nn.Sequential(*layers)
        optimizer['lr'] = optimizer.get('lr', 5e-4)
        optimizer_class = optimizer.pop('class')
        self.optimizer = optimizer_class(self.parameters(), **optimizer)
    
    
    def forward(self, data, deterministic=False):
        out = self.mlp(data)
        if self.distribution is None:
            return out
        elif self.output_transform in {"DiagGaussianDistribution", "SquashedDiagGaussianDistribution"}:
            mean, log_std = th.split(out, self.action_dim, dim=-1)
            return self.distribution.actions_from_params(mean_actions=mean, log_std=log_std, deterministic=deterministic)
        else:
            return self.distribution.actions_from_params(action_logits=out, deterministic=deterministic)


def set_method(x, old, new, **kwargs):
    old_method = getattr(x, old)
    kwargs["super_"] = old_method
    setattr(x, old, MethodType(partial(new, **kwargs), x))

def restore(self, original_methods, f=None):
    for name, method in original_methods:
        setattr(self, name, method)
    if f:
        f(self)

def set_restore(learner, f=None):
    """
    Add restore method
    """
    original_methods = inspect.getmembers(learner, predicate=inspect.ismethod)[1:]
    learner.restore = MethodType(partial(restore, original_methods=original_methods, f=f), learner)

def save(
    self,
    save_path: Union[str, Path, io.BufferedIOBase],
    super_,
    models: List[th.nn.Module],
    *args, **kwargs
) -> None:
    super_(save_path=save_path, *args, **kwargs)
    with ZipFile(save_path, mode="rw") as archive:
        for name, model in models.items():
            with archive.open('sbil_' + name + ".pth", mode="w") as f:
                th.save(model.state_dict(), f)
        archive.writestr("_stable_baselines3_version", sb.__version__ + "\nsbil")

def load(path, models):
    with ZipFile(save_path, mode="r") as archive:
        with archive.open("_stable_baselines3_version", mode="r") as f:
            lines = f.readlines()
        assert "sbil" in lines, "The learner that you are trying to load has never been saved with sbil."
        for name, model in models.items():
            with archive.open('sbil_' + name + '.pth', mode="r") as f:
                models.load_state_dict(th.load(f))

def ok(x):
    return x.pop('ok', True) in {True, None}

def get_class(str):
    try:
        return reduce(getattr, str.split("."), sys.modules[__name__])
    except AttributeError:
        return None

def make_config():
    parser = ArgumentParser(description='Training')
    parser.add_argument("-c", "--config", help="The configuration yaml file", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # import gym environment
    if config['env'].get('import', None) is not None:
        gym_import = importlib.import_module(config['env'].pop('import'))
    
    return config, gym_import

def make_env(config_env):
    config_env = {k.lower().strip(): v for k, v in config_env.items()}
    # gym environment wrappers
    max_episode_steps = config_env.pop('max_episode_steps', None)
    normalize = config_env.pop('normalize', None)
    vecenv = config_env.pop('vecenv', None)
    n_envs = config_env.pop('n_envs', None)
    if n_envs is None:
        n_envs = 1
    elif n_envs < 0:
        n_envs = os.cpu_count()
    
    time_limit_wrap = TimeLimitAware if config_env.pop('timelimitaware', None) else gym.wrappers.TimeLimit
    
    wrappers = {
        lambda env_: sbil.demo.AbsorbingState(env_): config_env.pop('absorbingstate', None),
        lambda env_: time_limit_wrap(env_, max_episode_steps=max_episode_steps): max_episode_steps,
    }
    def make_env_():
        env_ = gym.make(**config_env)
        for key, value in wrappers.items():
            if value:
                env_ = key(env_)
        return env_

    # Vectorized environment
    if vecenv is not None:
        vecenv_ = {'dummy':DummyVecEnv, 'subproc':SubprocVecEnv}[vecenv.lower()]
    env = vecenv_([make_env_ for  i in range(n_envs)])
    if normalize is not None:
        env = VecNormalize(env, **normalize)
    return env

def make_learner(config_learner, env, config_algorithm=None):
    learner_class_name = config_learner.pop('class')
    if hasattr(sb, learner_class_name):
        learner_class = getattr(sb, learner_class_name)
    elif importlib.util.find_spec("sb3_contrib"):
        import sb3_contrib
        if hasattr(sb3_contrib, learner_class_name):
            learner_class = getattr(sb, learner_class_name)
    else:
        learner_class = get_class(learner_class_name)
        assert learner_class is not None, f"Learner class names ({learner_class_name}) are not matching"
    
    policy_load = None
    if isinstance(config_learner['policy'], dict):
        policy_load = config_learner['policy'].get('load', None)
        config_learner['policy'] = config_learner['class']
    
    load = config_learner.pop('load', None)
    # convert str to float or int
    p = inspect.signature(learner_class).parameters
    for key, value in config_learner.items():
        if p[key].annotation in {float, int}:
            value = p[key].annotation(value)
    # Instanciate learner
    config_learner['env'] = env
    if load is None:
        learner = learner_class(**config_learner)
        if policy_load:
            learner.policy = learner.policy.load(policy_load)
    else:
        print(f"Loading learner {load}.")
        learner = learner_class.load(load, env=env)
    
    # algorithm
    il_algorithm = None
    if config_algorithm is not None and ok(config_algorithm):
        category = set(config_algorithm.keys()) & {"demo", "goal", "query", "custom"}
        assert len(category) == 1, "You must choose either demo, goal, query or custom in algorithm."
        category = next(iter(category))
        if category == "custom":
            il_algorithm = getattr(m, category)
        else:
            f = attrgetter(category + "." + config_algorithm.pop(category))
            il_algorithm = f(sbil)
        learner = il_algorithm(learner, **config_algorithm)
        config_algorithm['algorithm'] = il_algorithm.__name__
    
    return learner
