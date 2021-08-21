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
import shutil
from copy import deepcopy


import gym
from gym.spaces import Box
import pandas as pd
import numpy as np
import torch as th
from torch.nn import functional as F
import stable_baselines3 as sb
import sklearn as sk
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# try to import
try_import = {
    'seaborn': 'sns',
    'imageio': 'imageio',
    'pygifsicle': 'pygifsicle',
}
for k,v in try_import.items():
    if importlib.util.find_spec(k):
        globals()[v] = importlib.import_module(k)


import stable_baselines3 as sb
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.policies import BaseModel, get_policy_from_name
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.monitor import Monitor
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
from stable_baselines3.common.utils import polyak_update, get_device, get_latest_run_id
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, TD3, DQN

try:
    from sb3_contrib import TQC, QRDQN
    from sb3_contrib.common.utils import quantile_huber_loss
except ImportError:
    TQC, QRDQN = None, None

from sbil.data.generate_demo import generate_demo
import sbil
import sbil.demo

OnOrOff = Union[OffPolicyAlgorithm, OnPolicyAlgorithm]
Info = Dict[str, Any]

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
        super(TimeLimitAware, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

        if isinstance(self.observation_space, Box):
            assert len(self.observation_space.shape) == 1, "For Box spaces, 1D spaces are only supported."
            self.observation_space = Box(
                low=np.hstack((self.observation_space.low,-1)),
                high=np.hstack((self.observation_space.high,1)),
                dtype=self.observation_space.dtype,
            )
        else:
            raise NotImplementedError(f"TimeLimitAware does not support {self.observation_space}.")

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
        return self.observation(self.env.reset(**kwargs))


class MLP(th.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        net_arch: List[int] = [32,32],
        output_transform: Optional[Union[str, Callable[[], th.nn.Module]]] = None,
        spectral_norm: bool = False,
        optimizer: Dict[str, Any] = {},
    ) -> None:
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_arch = net_arch
        self.output_transform = output_transform
        self.optimizer_data = optimizer.copy()
        self.spectral_norm = spectral_norm
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
        optimizer_class = optimizer.pop('class', th.optim.Adam)
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

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        :param path:
        """
        th.save(
            {
                "state_dict": self.state_dict(),
                "data": {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "net_arch": self.net_arch,
                    "output_transform": self.output_transform,
                    "spectral_norm": self.spectral_norm,
                    "optimizer": self.optimizer_data,
                }
            },
            path
        )

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto") -> "MLP":
        """
        Load model from path.
        :param path:
        :param device: Device on which the network should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        model.load_state_dict(saved_variables["state_dict"])
        return model.to(device)

class EvalSaveGif(EvalCallback):
    """
    Evaluate the learner, save a gif and the learner of the last evaluation (overwrite).
    Gather all logs in the same folder.
    The logger is reconfigured to output csv as well.
    The a copy of the config file can be copied.
    """
    def __init__(self,
        eval_env,
        log_path,
        best_model_save_path,
        *args,
        period: int = 1,
        mode='rgb_array',
        **kwargs,
    ):
        super(EvalSaveGif, self).__init__(
            eval_env=eval_env,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            *args,
            **kwargs,
        )

        self.count = 0
        self.mode = mode
        self.period = period

    def _on_training_start(self): # setup the csv logger
        self.dir = self.logger.get_dir() or self.log_path

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
            fps = self.model.env.get_attr("metadata")[0].get('video.frames_per_second', 30) / self.period
            imageio.mimsave(self.dir+"/eval.gif", self.images, fps=fps)
            self.model.save(self.dir+"/last_model.zip")
            g = globals()
            if 'pygifsicle' in g: # optimize gif if possible
                g['pygifsicle'].optimize(self.dir+"/eval.gif")
            return out

        return True

def get_tensorboard_path(tensorboard_log, tb_log_name: str, reset_num_timesteps=True):
    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = tensorboard_log + f"/{tb_log_name}_{latest_run_id + 1}"
    else:
        save_path = None
    return save_path


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
    if isinstance(x, BaseAlgorithm) and old not in x._excluded_save_params():
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

def save_torch(
    self,
    path: Union[str, Path, io.BufferedIOBase],
    super_: Callable[..., None],
    modules: List[th.nn.Module],
    *args, **kwargs
) -> None:
    """
    Save method to save additional modules in the zip file.
    Save models with .pt extensions as .pth is reserved for stable baselines:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/save_util.py
    """
    super_(path=path, *args, **kwargs)
    with ZipFile(path, mode="a") as archive:
        for name, model in modules.items():
            with archive.open('sbil_' + name + ".pt", mode="w") as f:
                th.save(model.state_dict(), f)

def load_torch(path: str, modules: Dict[str, th.nn.Module]) -> None:
    """
    Load additional modules in the zip file.
    """
    with ZipFile(path, mode="r") as archive:
        assert {'sbil_'+k+".pt" for k in modules} < set(archive.namelist()), ("Failed to load"
            "Some modules are missing. Make sure you saved the model with sbil."
        )
        for name, model in modules.items():
            with archive.open('sbil_' + name + '.pt', mode="r") as f:
                model.load_state_dict(th.load(f))

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
        learner = learner_class.load(load, env=env)
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

def get_policy(learner: BaseAlgorithm) -> BasePolicy:
    if hasattr(learner, "actor") and learner.actor is not None:
        return learner.actor
    else:
        return learner.policy

def get_features_extractor(learner: BaseAlgorithm) -> BasePolicy:
    """
    Return the object having the extract_features() and _predict method.
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

def action_loss(policy: BasePolicy, observations: th.Tensor, actions: th.Tensor) -> th.Tensor:
    """
    Return the action loss given the observations as the input and and the
    actions as the target. The loss is:
    - mse_loss if the policy is deterministic like TD3
    - cross_entropy if the policy is the Q value, like DQN
    - log_prob otherwise, like SAC or PPO
    The returned loss is not reduced with size (-1,1)

    :param policy: policy obtained with get_policy()
    :param observations: observations to pass to the policy
    :param actions: Target action scaled to [-1, 1] if continuous (ReplayBuffer)
    """
    q_net = getattr(policy, "q_net", getattr(policy, "quantile_net", None))

    if hasattr(policy, "evaluate_actions"): # PPO
        value, log_prob, entropy = policy.evaluate_actions(observations, actions)
        loss = -log_prob
    # actor
    elif hasattr(policy, "action_dist"): # SAC, TQC
        # first pass observations to set the distribution
        policy(observations)
        loss = -policy.action_dist.log_prob(actions)
    elif hasattr(policy, "mu"): # TD3
        loss = F.mse_loss(input=policy(observations), target=actions, reduction='none').mean(-1)
    # Q value policy
    elif q_net is not None: # DQN, QRDQN
        input = q_net(observations)
        if hasattr(policy, "quantile_net"):
            input = input.mean(1)
        loss = F.cross_entropy(input=input, target=actions.squeeze(), reduction='none')
    return loss.view(-1, 1)

def scale_action(action: np.ndarray, space) -> np.ndarray:
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action: Action to scale
    :return: Scaled action
    """
    if not isinstance(space, gym.spaces.Box):
        return action
    low, high = space.low, space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0

def unscale_action(scaled_action: np.ndarray, space) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param scaled_action: Action to un-scale
    """
    if not isinstance(space, gym.spaces.Box):
        return scaled_action
    low, high = space.low, space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))

def add_metric(x: Dict[str, List[float]], y: Dict[str, float] = None) -> None:
    if y is None: return
    for k, v in y.items():
        x[k] = (x[k] + [v]) if k in x else [v]

def update_learning_rate(self):
    if hasattr(self, "q_net") or hasattr(self, "quantile_net"):
        optimizers = [self.policy.optimizer]
    else:
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if getattr(self, "ent_coef_optimizer", None) is not None:
            optimizers += [self.ent_coef_optimizer]
    self._update_learning_rate(optimizers)

def entropy(self, data):
    if hasattr(self, "ent_coef_optimizer"):
        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(data['replay'].observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        else:
            ent_coef = self.ent_coef_tensor

        data['ent_coef'] = ent_coef
        data['actions_pi'] = actions_pi
        data['log_prob'] = log_prob

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            return {'ent_coef_losses': ent_coef_loss.item(), 'ent_coef': ent_coef.item()}
        else:
            return {'ent_coef': ent_coef.item()}

    return {}



def critic_loss(self, data):
    replay_data = data['replay']
    if isinstance(self, SAC):
        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - data['ent_coef'] * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        data['current_q_values'] = current_q_values

    elif isinstance(self, TD3):
        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        data['current_q_values'] = current_q_values

    elif isinstance(self, DQN):
        with th.no_grad():
            # Compute the next Q-values using the target network
            next_q_values = self.q_net_target(replay_data.next_observations)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates
        current_q_values = self.q_net(replay_data.observations)
        data['current_q_values_all'] = current_q_values
        # Retrieve the q-values for the actions from the replay buffer
        current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
        data['current_q_values'] = current_q_values
        # Compute Huber loss (less sensitive to outliers)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        data['loss'] = loss
        return {'loss': loss.item()}

    elif TQC is not None and isinstance(self, TQC):
        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute and cut quantiles at the next state
            # batch x nets x quantiles
            next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

            # Sort and drop top k quantiles to control overestimation.
            n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
            next_quantiles, _ = th.sort(next_quantiles.reshape(data['batch_size'], -1))
            next_quantiles = next_quantiles[:, :n_target_quantiles]

            # td error + entropy term
            target_quantiles = next_quantiles - data['ent_coef'] * next_log_prob.reshape(-1, 1)
            target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
            # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
            target_quantiles.unsqueeze_(dim=1)

        # Get current Quantile estimates using action from the replay buffer
        current_quantiles = self.critic(replay_data.observations, replay_data.actions)
        # Compute critic loss, not summing over the quantile dimension as in the paper.
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
        data['current_quantiles'] = current_quantiles

    elif QRDQN is not None and isinstance(self, QRDQN):
        with th.no_grad():
            # Compute the quantiles of next observation
            next_quantiles = self.quantile_net_target(replay_data.next_observations)
            # Compute the greedy actions which maximize the next Q values
            next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1)
            next_greedy_actions = next_greedy_actions.expand(data['batch_size'], self.n_quantiles, 1)
            # Follow greedy policy: use the one with the highest Q values
            next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            # 1-step TD target
            target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_quantiles

        # Get current quantile estimates
        current_quantiles = self.quantile_net(replay_data.observations)
        data['current_quantiles_all'] = current_quantiles
        # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1).
        actions = replay_data.actions[..., None].long().expand(data['batch_size'], self.n_quantiles, 1)
        # Retrieve the quantiles for the actions from the replay buffer
        current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)
        data['current_quantiles'] = current_quantiles
        # Compute Quantile Huber loss, summing over a quantile dimension as in the paper.
        loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)

        data['loss'] = loss
        return {'loss': loss.item()}
    else:
        raise NotImplementedError(f"critic_loss is not supported for {type(self).__name__}.")

    data['critic_loss'] = critic_loss
    return {'critic_loss': critic_loss.item()}

def optimize_critic(self, data):
    if hasattr(self, "q_net") or hasattr(self, "quantile_net"):
        self.policy.optimizer.zero_grad()
        data['loss'].backward()
        # Clip gradient norm
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
    else:
        self.critic.optimizer.zero_grad()
        data['critic_loss'].backward()
        self.critic.optimizer.step()
        if (
            (hasattr(self, "target_update_interval") and data['gradient_step'] % self.target_update_interval == 0)
            or (hasattr(self, "policy_delay") and self._n_updates % self.policy_delay == 0)
        ):
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

def actor_loss(self, data):
    replay_data = data['replay']
    if isinstance(self, TD3):
        qf_pi = self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations))
        actor_loss = -qf_pi.mean()
    elif isinstance(self, SAC):
        q_values_pi = th.cat(self.critic.forward(replay_data.observations, data['actions_pi']), dim=1)
        qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (data['ent_coef'] * data['log_prob'] - qf_pi).mean()
    elif TQC is not None and isinstance(self, TQC):
        qf_pi = self.critic(replay_data.observations, data['actions_pi']).mean(dim=2).mean(dim=1, keepdim=True)
        actor_loss = (data['ent_coef'] * data['log_prob'] - qf_pi).mean()
    elif hasattr(self, "q_net") or hasattr(self, "quantile_net"):
        return {}
    else:
        raise NotImplementedError(f"actor_loss does not support {type(self).__name__}.")

    data['qf_pi'] = qf_pi
    data['actor_loss'] = actor_loss
    return {'actor_loss': actor_loss.item()}

def optimize_actor(self, data):
    if 'actor_loss' in data and self.actor is not None:
        self.actor.optimizer.zero_grad()
        data['actor_loss'].backward()
        self.actor.optimizer.step()
        if hasattr(self, "policy_delay") and self._n_updates % self.policy_delay == 0:
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

def record(self, metrics, data):
    if 'actor_loss' in metrics:
        self.logger.record("train/actor_loss", np.mean(metrics['actor_loss']))
        self.logger.record("train/critic_loss", np.mean(metrics['critic_loss']))
    elif 'loss' in metrics:
        self.logger.record("train/loss", np.mean(metrics['loss']))
    if isinstance(self, SAC) or (TQC is not None and isinstance(self, TQC)):
        self.logger.record("train/ent_coef", np.mean(metrics['ent_coef']))
        if 'ent_coef_loss' in metrics:
            self.logger.record("train/ent_coef_loss", np.mean(metrics['ent_coef_loss']))


def train_off(
    self,
    gradient_steps: int,
    batch_size: int = 64,
    update_learning_rate=update_learning_rate,
    begin: Callable[[OnOrOff, Info], Optional[Info]] = lambda learner, data: None,
    entropy=entropy,
    critic_loss=critic_loss,
    optimize_critic=optimize_critic,
    actor_loss=actor_loss,
    optimize_actor=optimize_actor,
    end: Callable[[OnOrOff, Info], Optional[Info]] = lambda learner, data: None,
    super_=None,
    *args,
    **kwargs
) -> None:

    update_learning_rate(self)

    metrics = {} # tensorboard metrics
    data = {'batch_size': batch_size} # user data

    for gradient_step in range(gradient_steps):
        data['gradient_step'] = gradient_step
        self._n_updates += 1
        data['replay'] = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        if self.use_sde:
            self.actor.reset_noise()

        add_metric(metrics, begin(self, data)) # begin setup

        add_metric(metrics, entropy(self, data))

        add_metric(metrics, critic_loss(self, data))
        optimize_critic(self, data)

        add_metric(metrics, actor_loss(self, data))
        optimize_actor(self, data)

        add_metric(metrics, end(self, data)) # end setup

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    record(self, metrics, data)
