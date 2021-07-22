from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pathlib
import io
import os, sys
from argparse import ArgumentParser
import yaml
import importlib
from types import MethodType
from functools import partial, reduce
import functools
import inspect
from zipfile import ZipFile
from operator import attrgetter



import gym
from gym.spaces import Box
import pandas as pd
import numpy as np
import torch as th
import stable_baselines3 as sb
import sklearn as sk
import matplotlib.pyplot as plt

# try to import
try_import = {
    'seaborn': 'sns',
    'imageio': 'imageio'
}
for k,v in try_import.items():
    if importlib.util.find_spec(k):
        globals()[v] = importlib.import_module(k)


import stable_baselines3 as sb
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.policies import BaseModel, get_policy_from_name
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import (
DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    CategoricalDistribution,
    MultiCategoricalDistribution,
    BernoulliDistribution,
    StateDependentNoiseDistribution
)
from stable_baselines3.common.vec_env import (
    VecVideoRecorder,
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize
)
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, ConvertCallback
from stable_baselines3.common.logger import Logger, CSVOutputFormat, configure, Video
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv

from sbil.data.generate_demo import generate_demo
import sbil


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
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        net_arch: List[int] = [32,32],
        output_transform: Optional[Union[str, Callable[[], th.nn.Module]]] = None,
        spectral_norm: bool = False,
        optimizer: Dict[str, Any] = {'class':th.optim.Adam},
    ) -> None:
        super(MLP, self).__init__()

        self.net_arch = net_arch
        self.output_transform = output_transform
        ot = output_transform
        distributions = {
            "DiagGaussianDistribution": DiagGaussianDistribution,
            "SquashedDiagGaussianDistribution": SquashedDiagGaussianDistribution,
            "CategoricalDistribution": CategoricalDistribution,
            "MultiCategoricalDistribution": MultiCategoricalDistribution,
            "BernoulliDistribution": BernoulliDistribution,
            "StateDependentNoiseDistribution": StateDependentNoiseDistribution,
        }
        self.double_dim = output_transform in {
            "DiagGaussianDistribution",
            "SquashedDiagGaussianDistribution"
        }
        if ot in distributions:
            self.distribution = distributions[ot](self.action_dim)
            if self.double_dim:
                output_dim = self.action_dim*2
        else:
            self.distribution = None

        layers = create_mlp(input_dim=input_dim,
            output_dim=output_dim,
            net_arch=net_arch,
            activation_fn=th.nn.LeakyReLU
        )
        if spectral_norm:
            i = 0
            for layer in layers:
                if isinstance(layer, th.nn.Linear) and i < len(net_arch):
                    i += 1
                    layer = th.nn.utils.spectral_norm(layer) # use parametrizations
        if ot is not None and ot not in distributions:
            layers += [getattr(th.nn, ot)() if isinstance(ot, str) else ot()]
        self.mlp = th.nn.Sequential(*layers)
        optimizer['lr'] = optimizer.get('lr', 5e-4)
        optimizer_class = optimizer.pop('class')
        self.optimizer = optimizer_class(self.mlp.parameters(), **optimizer)


    def forward(self, data, deterministic=False):
        out = self.mlp(data)
        if self.distribution is None:
            return out
        elif self.double_dim:
            mean, log_std = th.split(out, self.action_dim, dim=-1)
            kwargs = dict(mean_actions=mean, log_std=log_std)
        else:
            kwargs = dict(action_logits=out)
        self.distribution.actions_from_params(**kwargs, deterministic=deterministic)

class EvalSaveGif(EvalCallback):
    """
    Evaluate the learner, save a gif and the learner of the last evaluation (overwrite).
    The logger is reconfigured to output csv as well.
    """
    def __init__(self, eval_env, *args, period: int = 1, mode='rgb_array', **kwargs):
        super(EvalSaveGif, self).__init__(eval_env=eval_env, *args, **kwargs)
        self.count = 0
        self.mode = mode
        self.period = period

    def _on_training_start(self): # setup the csv logger
        self.dir = self.logger.get_dir() or self.log_path
        logger = configure(
            folder=self.dir,
            format_strings=(
                ['csv', 'tensorboard'] if self.model.tensorboard_log is not None
                else ['csv']
            )
        )
        self.model.set_logger(logger) # set logger to the model
        self.logger = logger # set logger to the callback

    def _log_success_callback(self, locals_, globals_) -> None:
        super()._log_success_callback(locals_, globals_)
        if self.count%self.period == 0:
            self.images.append(self.eval_env.render(mode=self.mode)) #.transpose(2, 0, 1)
        self.count += 1

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.count = 0
            self.images = []
            out = super()._on_step()
            fps = self.model.env.metadata.get('video.frames_per_second', 30) / self.period
            imageio.mimsave(self.dir+"/eval.gif", self.images, fps=fps)
            self.model.save(self.dir+"/last_model.zip")
            return out
        return True

def _excluded_save_params(
    self,
    super_: Callable[[], List[str]],
    additionals: List[str],
) -> List[str]:
    """Method, exclude additional attributes. """
    return super_() + additionals

def set_method(x, old:str, new: Callable[..., Any], **kwargs):
    """
    Set a new method and overwrite the old one.
    """
    old_method = getattr(x, old)
    kwargs["super_"] = old_method
    setattr(x, old, MethodType(partial(new, **kwargs), x))
    if isinstance(x, BaseAlgorithm):
        # exclude the method for saving
        old_exclude = x._excluded_save_params
        x._excluded_save_params = MethodType(
            partial(
                _excluded_save_params,
                super_=old_exclude,
                additionals=[old]
            ), x
        )

def restore(
    self,
    original_methods: List[Tuple[str, MethodType]],
    f: Optional[Callable[Any, Any]] = None,
) -> None:
    """Method, restore all methods and do f."""
    for name, method in original_methods:
        setattr(self, name, method)
    if f:
        f(self)

def set_restore(x, f: Callable[Any, Any] = None) -> None:
    """
    Add restore method
    """
    original_methods = inspect.getmembers(x, predicate=inspect.ismethod)[1:]
    x.restore = MethodType(partial(restore, original_methods=original_methods, f=f), x)
    if isinstance(x, BaseAlgorithm):
        set_method(
            x,
            "_excluded_save_params",
            _excluded_save_params,
            additionals=[ # overwritten methods that should not be saved
                "restore",
                "_excluded_save_params",
                "save",
                "save_replay_buffer",
                "train",
            ]
        )
        if isinstance(x, OffPolicyAlgorithm):
            set_method(x, "save_replay_buffer", save_replay_buffer)

def save_replay_buffer(
    self,
    path: Union[str, pathlib.Path, io.BufferedIOBase],
    super_: Callable[[str], None]
) -> None:
    """
    Method, save a copied replay buffer.
    """
    # copy the buffer
    old_buffer = self.replay_buffer
    new_buffer = type(old_buffer)(
        buffer_size=old_buffer.buffer_size,
        observation_space=old_buffer.observation_space,
        action_space=old_buffer.action_space,
        device=old_buffer.device,
        n_envs=old_buffer.n_envs,
        optimize_memory_usage=old_buffer.optimize_memory_usage,
        handle_timeout_termination=old_buffer.handle_timeout_termination,
    )
    is_dict = isinstance(old_buffer.observations, dict)
    copy_obs = (lambda obs: obs.copy()) if is_dict else lambda obs: np.array(obs)
    new_buffer.observations = copy_obs(old_buffer.observations)
    new_buffer.actions = np.array(old_buffer.actions)
    new_buffer.rewards = np.array(old_buffer.rewards)
    new_buffer.dones = np.array(old_buffer.dones)
    new_buffer.timeouts = np.array(old_buffer.timeouts)
    new_buffer.pos = old_buffer.pos
    new_buffer.full = old_buffer.full
    if not old_buffer.optimize_memory_usage:
        new_buffer.next_observations = copy_obs(old_buffer.next_observations)
    # save the copied because methods and attributes are original
    self.replay_buffer = new_buffer
    super_(path=path)
    self.replay_buffer = old_buffer

def save(
    self,
    path: Union[str, Path, io.BufferedIOBase],
    super_: Callable[..., None],
    modules: List[th.nn.Module],
    *args, **kwargs
) -> None:
    """ Save method to save additional modules in the zip file."""
    super_(path=path, *args, **kwargs)
    with ZipFile(path, mode="a") as archive:
        for name, model in modules.items():
            with archive.open('sbil_' + name + ".pth", mode="w") as f:
                th.save(model.state_dict(), f)

def load(path: str, modules: Dict[str, th.nn.Module]) -> None:
    """
    Load additional modules in the zip file.
    """
    with ZipFile(save_path, mode="r") as archive:
        assert set(modules.keys()) < archive.namelist(), ("Failed to load"
            "Some modules are missing. Make sure you saved the model with sbil."
        )
        for name, model in modules.items():
            with archive.open('sbil_' + name + '.pth', mode="r") as f:
                models.load_state_dict(th.load(f))

def ok(x: Dict[str, Any]) -> bool:
    return x.pop('ok', True) in {True, None}

def get_class(class_name: str) -> Union[type[Any], None]:
    """ Convert string to class, return None if not found."""
    try:
        return reduce(getattr, class_name.split("."), sys.modules[__name__])
    except AttributeError:
        return None

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
    if config['env'].get('import', None) is not None:
        gym_import = importlib.import_module(config['env'].pop('import'))
    else:
        gym_import = None

    return config, gym_import

def make_env(config_env: Dict[str, str]) -> GymEnv:
    config_env = {k.lower().strip(): v for k, v in config_env.items()}
    # gym environment wrappers
    max_steps = config_env.pop('max_episode_steps', None)
    absorb = config_env.pop('absorbingstate', None)
    normalize = config_env.pop('normalize', None)
    vecenv = config_env.pop('vecenv', None)
    n_envs = config_env.pop('n_envs', None)
    if n_envs is None:
        n_envs = 1
    elif n_envs < 0:
        n_envs = os.cpu_count()

    if config_env.pop('timelimitaware', None):
        time_wrap = TimeLimitAware
    else:
        time_wrap = gym.wrappers.TimeLimit

    wrappers = {
        lambda env_: sbil.demo.AbsorbingState(env_): absorb,
        lambda env_: time_wrap(env_, max_episode_steps=max_steps): max_steps,
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
    else:
        vecenv_ = DummyVecEnv
    env = vecenv_([make_env_ for  i in range(n_envs)])
    if normalize is not None:
        env = VecNormalize(env, **normalize)
    return env

def make_learner(
    config_learner:Dict[str, str],
    env: GymEnv,
    config_algorithm: Dict[str, str]=None
) -> BaseAlgorithm:
    learner_class_name = config_learner.pop('class')
    if hasattr(sb, learner_class_name):
        learner_class = getattr(sb, learner_class_name)
    elif importlib.util.find_spec("sb3_contrib"):
        import sb3_contrib
        if hasattr(sb3_contrib, learner_class_name):
            learner_class = getattr(sb3_contrib, learner_class_name)
    else:
        learner_class = get_class(learner_class_name)
        assert learner_class is not None, (
            f"Learner class names ({learner_class_name}) are not matching"
        )

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
        assert len(category) == 1, (
            "You must choose either demo, goal, query or custom in algorithm."
        )
        category = next(iter(category))
        if category == "custom":
            il_algorithm = getattr(m, category)
        else:
            f = attrgetter(category + "." + config_algorithm.pop(category))
            il_algorithm = f(sbil)
        if category == "demo": # generate demo_buffer is needed
            if config_algorithm.get('demo_buffer', None) is None:
                config_algorithm['demo_buffer'] = generate_demo(env)
        learner = il_algorithm(learner, **config_algorithm)
        config_algorithm['algo'] = il_algorithm.__name__

    return learner

def get_policy(learner: BaseAlgorithm) -> BasePolicy:
    """
    Get policy from learner. The returned policy has the extract_features() method.
    """
    if isinstance(learner, OnPolicyAlgorithm): # PPO, A2C
        return learner.policy
    # single network
    elif hasattr(learner, "q_net"): # DQN
        return learner.q_net
    elif hasattr(learner, "quantile_net"): # QRDQN
        return learner.quantile_net
    # double network
    else: # SAC, TQC, TD3, DDPG
        return learner.actor
