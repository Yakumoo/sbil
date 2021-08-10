
import argparse
import yaml
from pathlib import Path
import importlib
import gym
from functools import partial
import os
from inspect import signature, ismethod
import sys
import ast
from operator import attrgetter
import shutil

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import sbil
import sbil.demo
import sbil.goal
import stable_baselines3 as sb
from sbil.utils import safe_eval, make_config, make_env, make_learner, ok, EvalSaveGif, get_class
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

# for multiprocessing, the import must be in the global scope
config, config_path, gym_import = make_config()
assert {"env", "learner"} <= set(config.keys()), "env and learner are required in the yaml file."

class CopyConfigCallback(BaseCallback):
    """ Copy the config file in the log folder. """
    def __init__(self, config_path, verbose=0):
        super(CopyConfigCallback, self).__init__(verbose)
        self.config_path = config_path

    def _on_training_start(self):
        shutil.copyfile(self.config_path, self.logger.get_dir() + "/config.yaml")

    def _on_step(self) -> bool:
        return True

def main():

    # unpack
    config_env = config['env'].copy()
    config_learner = config['learner']
    config_algo = config.get('algorithm', None)
    config_learn = config.get('learn', None)
    config_declare = config.get('declare', None)
    config_save = config.get('save', None)

    # execute custom python code
    if config_declare is not None:
        config_declare = config_declare.strip()
        if config_declare[-3:] == ".py":
            print(f"Executing {config_declare}, use codes that you trust!")
            with open(config_declare) as f: # https://stackoverflow.com/a/437857
                code = compile(f.read(), config_declare, 'exec')
                exec(code, {}, {}) # allow import, safe ?
        else:
            print("Evaluating declare")
            safe_eval(config_declare) # import forbidden
    m = sys.modules[__name__]

    # gym environment wrappers
    env = make_env(config_env)
    # learner
    learner = make_learner(config_learner, env, config_algo)
    print(
        f"Using {type(learner).__name__}" + (f" and {config_algo['algo']}"
        if config_algo and 'algo' in config_algo else "")
        +f" on {config_env['id']}"
    )
    is_off = isinstance(learner, OffPolicyAlgorithm)

    # try add csv to logger
    tensorboard_log = config_learner.get('tensorboard_log', None)
    tb_log_name = config_learn.get('tb_log_name', None) or ''
    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None and not learner._custom_logger:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not (config_learn.get('reset_num_timesteps', None) or True):
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = tensorboard_log + f"/{tb_log_name}_{latest_run_id + 1}"
        learner.set_logger(configure(folder=save_path, format_strings=['csv', 'tensorboard']))

    if config_learn is not None and ok(config_learn):
        # callback
        if config_learn.get('callback', None) is not None:
            callback_dict = config_learn.pop('callback')
            assert 'class' in callback_dict, "You must specify class in learn->callback."

            callback_class_name = callback_dict.pop('class')
            callback_class = get_class(callback_class_name)
            if callback_class is None:
                print("You are calling a callback that doesn't exist in learn")
                return

            # add env if needed
            callback_param = signature(callback_class).parameters
            if 'eval_env' in callback_param:
                callback_dict['eval_env'] = env

            callback = [CopyConfigCallback(config_path), callback_class(**callback_dict)]

        elif tensorboard_log is not None:
            callback = CopyConfigCallback(config_path)
        else: callback = None

        learner.learn(**config_learn, callback=callback)

    if config_save is not None and ok(config_save):
        if config_save.get('learner', None) is not None:
            learner.save(config_save['learner'])
        if config_save.get('policy', None) is not None:
            learner.policy.save(config_save['policy'])
        if config_save.get('buffer', None) is not None and is_off:
            learner.save_replay_buffer(config_save['buffer'])
        if config_save.get('env', None) is not None and hasattr(env, "save"):
            env.save(config_save['env'])


if __name__ == "__main__":
    main()
