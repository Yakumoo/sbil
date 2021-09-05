from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from argparse import ArgumentParser
from functools import partial, reduce
import inspect
from copy import deepcopy
from operator import attrgetter
import importlib
import sys

import yaml
import gym
#from gym.spaces import Box
import pandas as pd
import numpy as np
import torch as th
import stable_baselines3 as sb
import sklearn as sk
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import (
    VecVideoRecorder,
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize
)

from sbil.utils import TimeLimitAware, EvalSaveGif
import sbil.demo

# try to import
try_import = {
    'seaborn': 'sns',
    'imageio': 'imageio',
    'pygifsicle': 'pygifsicle',
}
for k,v in try_import.items():
    if importlib.util.find_spec(k):
        globals()[v] = importlib.import_module(k)

def safe_eval(s:str) -> None:
    """ Excecute a string code using a whitelist. """
    global_symbols = { # whitelist
        # available builtins
        # removed: eval, __import__
        '__builtins__': {k: __builtins__[k] for k in ['True', 'False', 'None', 'Ellipsis', "abs", "delattr", "hash", "memoryview", "set", "all", "dict", "help", "min", "setattr", "any", "dir", "hex", "next", "slice", "ascii", "divmod", "id", "object", "sorted", "bin", "enumerate", "input", "oct", "staticmethod", "bool", "int", "open", "str", "breakpoint", "exec", "isinstance", "ord", "sum", "bytearray", "filter", "issubclass", "pow", "super", "bytes", "float", "iter", "print", "tuple", "callable", "format", "len", "property", "type", "chr", "frozenset", "list", "range", "vars", "classmethod", "getattr", "locals", "repr", "zip", "compile", "globals", "map", "reversed", "complex", "hasattr", "max", "round"]},
        # available modules
        'np': np,
        'th': th,
        'sb': sb,
        'pd': pd,
        'gym': gym,
        'sk': sk,
        'plt': plt,
    }
    global_symbols.update({v:globals()[v] for k, v in try_import.items() if v in globals()})
    return exec(s, global_symbols)


def ok(x: Dict[str, Any]) -> bool:
    return x.pop('ok', True) in {True, None}

def get_class(class_name: str) -> Union[type[Any], None]:
    """ Convert string to class, return None if not found."""
    try:
        return reduce(getattr, class_name.split("."), sys.modules[__name__])
    except AttributeError:
        return None


def clean_keys(x):
    if isinstance(x, dict):
        return {k.lower().strip(): clean_keys(v) for k, v in x.copy().items()}
    else:
        return x

def make_config() -> Tuple[Dict[str, Any], Any]:
    parser = ArgumentParser(description='Training')
    parser.add_argument(
        "-c", "--config",
        help="The configuration yaml file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # import gym environment
    gym_package = config['env'].get('gym package', config['env'].get('gym_package', None))
    if gym_package is not None:
        gym_package = importlib.import_module(gym_package)
    else:
        gym_package = None

    return config, args.config, gym_package

def wrapper_env(env, wrappers):
    """ wrap the env with wrappers """
    for key, value in wrappers.items():
        if value:
            env = key(env)
    return env

def make_env(config_env: Dict[str, str]) -> GymEnv:
    config_env_ = {k.lower().strip(): v for k, v in config_env.copy().items()}
    config_env_.pop('gym package', config_env_.pop('gym_package', None))
    # gym environment wrappers
    max_episode_steps = config_env_.pop('max_episode_steps', None)
    absorb = config_env_.pop('AbsorbingState', config_env_.pop('absorbingstate', config_env_.pop('absorbing_state', None)))
    normalize = config_env_.pop('normalize', config_env_.pop('Normalize', None))
    vecenv = config_env_.pop('VecEnv', config_env_.pop('vecenv', config_env_.pop('vec_env', 'dummy')))
    n_envs = config_env_.pop('n_envs', None)
    monitor = config_env_.pop('monitor', None)

    if n_envs is None:
        n_envs = 1
    elif n_envs < 0:
        n_envs = os.cpu_count()

    if config_env_.pop('timelimitaware', config_env_.pop('TimeLimitAware', config_env_.pop('time_limit_aware', None))):
        time_wrap = TimeLimitAware
    else:
        time_wrap = gym.wrappers.TimeLimit

    # available wrappers
    wrappers = {
        lambda env_: sbil.demo.AbsorbingState(env_): absorb,
    }
    vec_wrappers = {
        lambda env_: VecNormalize(env, **normalize): normalize,
    }

    # function to create the env, it is important to wrap the TimeLimit before Monitor
    if max_episode_steps:
        env_id = lambda: time_wrap(gym.make(**config_env_), max_episode_steps=max_episode_steps)
    else:
        env_id = lambda: gym.make(**config_env_)

    vec_env_cls = {'dummy':DummyVecEnv, 'subproc':SubprocVecEnv}[vecenv.lower()]

    if monitor is None: # wrap without the monitor
        env = vec_env_cls([lambda: wrapper_env(env_id(), wrappers) for i in range(n_envs)])
    else:
        env = make_vec_env(
            env_id=env_id,
            n_envs=n_envs,
            seed=None,
            start_index=0,
            monitor_dir=monitor.pop('dir', None),
            wrapper_class=wrapper_env,
            env_kwargs=None,#config_env_,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=None,
            monitor_kwargs=monitor,
            wrapper_kwargs={'wrappers': wrappers},
        )

    # final vec wrappers
    return wrapper_env(env, vec_wrappers)


def make_learner(
    config_learner:Dict[str, str],
    env: GymEnv,
    config_algorithm: Dict[str, str]=None
) -> BaseAlgorithm:

    learner_class_name = config_learner.pop('class')
    learner_class = None

    if hasattr(sb, learner_class_name):
        learner_class = getattr(sb, learner_class_name)
    elif importlib.util.find_spec("sb3_contrib"):
        import sb3_contrib
        if hasattr(sb3_contrib, learner_class_name):
            learner_class = getattr(sb3_contrib, learner_class_name)

    if learner_class is None:
        learner_class = get_class(learner_class_name)
        assert learner_class is not None, (
            f"Learner class names ({learner_class_name}) are not matching"
        )

    policy_load = None
    if isinstance(config_learner['policy'], dict):
        policy_load = config_learner['policy'].get('load', None)
        config_learner['policy'] = config_learner['class']

    load = config_learner.pop('load', None)
    load_replay_buffer = config_learner.pop('load_replay_buffer', None)

    # convert str to float or int
    p = inspect.signature(learner_class).parameters
    for key, value in config_learner.items():
        if p[key].annotation in {float, int}:
            value = p[key].annotation(value)

    # Instanciate learner
    if load is None:
        learner = learner_class(**config_learner, env=env)
        if policy_load:
            learner.policy = learner.policy.load(policy_load)
    else:
        print(f"Loading learner {load}.")
        learner = learner_class.load(load, env=env, device=config_learner.get('device', None) or 'auto')
        learner._last_obs = None # this will reset the environment

    # algorithm
    il_algorithm = None
    if config_algorithm is not None and ok(config_algorithm):
        category = set(config_algorithm.keys()) & {"demo", "goal", "query", "custom"}
        assert len(category) == 1, (
            "You must choose either demo, goal, query or custom in algorithm."
        )
        category = next(iter(category))
        algorithm = config_algorithm.pop(category)
        if category == "custom":
            il_algorithm = getattr(m, category)
        else:
            assert algorithm is not None and hasattr(getattr(sbil, category), algorithm), f"The algorithm you gave is incorrect {category}: {algorithm}"
            f = attrgetter(category + "." + algorithm)
            il_algorithm = f(sbil)

        # signature parameters of the IL algorithm
        p = inspect.signature(il_algorithm).parameters

        # generate demo_buffer if needed
        if "demo_buffer" in p:
            demo_buffer = config_algorithm.get('demo_buffer', None) or {}
            if isinstance(demo_buffer, dict):
                config_algorithm['demo_buffer'] = generate_demo(env, **demo_buffer)

        # additinal loading for the IL algorithm
        load = config_algorithm.get('load', load)
        if "load" in p and load is not None:
            config_algorithm['load'] = load

        learner = il_algorithm(learner, **config_algorithm)
        config_algorithm['algo'] = il_algorithm.__name__

    # load_replay_buffer
    if load_replay_buffer is not None:
        if load_replay_buffer is True:
            demo_buffer = config_algorithm.get('demo_buffer', None)
            assert config_algorithm is not None and demo_buffer, (
                "The replay buffer can not be deduced during load_replay_buffer, please indicate a path."
            )
            if isinstance(demo_buffer, str):
                learner.load_replay_buffer(demo_buffer)
            else:
                learner.replay_buffer = deepcopy(demo_buffer)
        else:
            learner.load_replay_buffer(load_replay_buffer)

    return learner
