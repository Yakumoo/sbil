
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

import sbil
import sbil.demo
import sbil.goal
import stable_baselines3 as sb
from sbil.utils import safe_eval, make_config, make_env, make_learner, ok, EvalSaveGif, get_class, get_tensorboard_path, clean_keys

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EventCallback,
    CallbackList,
    CheckpointCallback,
    ConvertCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    StopTrainingOnMaxEpisodes,
)

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

    # get tensorboard folder
    save_path = get_tensorboard_path(
        tensorboard_log=config_learner.get('tensorboard_log', None),
        tb_log_name=config_learn.get('tb_log_name', None) or '',
        reset_num_timesteps=config_learn.get('reset_num_timesteps', True),
    )

    # gym environment wrappers
    env = make_env(config_env)

    # learner
    learner = make_learner(config_learner, env, config_algo)
    learner.set_logger(configure(folder=save_path, format_strings=['csv', 'tensorboard']))
    is_off = isinstance(learner, OffPolicyAlgorithm)

    print(
        f"Using {type(learner).__name__}" + (f" and {config_algo['algo']}"
        if config_algo and 'algo' in config_algo else "")
        +f" on {config_env['id']}"
    )

    # add monitor for evaluation
    config_env['monitor'] = config_env.get('monitor', None) or ({'dir': save_path} if save_path else {})


    if config_learn is not None and ok(config_learn):
        # callback
        if config_learn.get('callback', None) is not None:
            callback_dict = config_learn.pop('callback')
            callback = [CopyConfigCallback(config_path)]

            for callback_class_name, callback_kwargs in callback_dict.items():
                callback_class = get_class(callback_class_name)
                assert callback_class, f"You are calling a callback {callback_class_name} that doesn't exist."

                # add parameters if needed
                callback_param = signature(callback_class).parameters
                if 'eval_env' in callback_param:
                    callback_kwargs['eval_env'] = make_env(config_env)
                if 'log_path' in callback_param:
                    callback_kwargs['log_path'] = callback_kwargs.get('log_path', None) or save_path
                if 'best_model_save_path' in callback_param:
                    callback_kwargs['best_model_save_path'] = callback_kwargs.get('best_model_save_path', None) or save_path

                callback.append(callback_class(**callback_kwargs))

        else:
            config_learn['eval_env'] = make_env(config_env)
            config_learn['eval_log_path'] = config_learn.get('eval_log_path', None) or save_path
            callback = None if save_path is None else CopyConfigCallback(config_path)

        # pretain with a behavioural cloning initialization
        pretrain = config_learn.pop('pretrain', None)
        if pretrain is not None:
            demo_buffer = (None if config_algo is None else config_algo.get('demo_buffer', None)) or pretrain.get('demo_buffer', None)
            assert demo_buffer, (
                "The demo_buffer for the pretraining can not be deduced, please indicate demo_buffer"
            )
            sbil.demo.bc(
                learner,
                gradient_steps=int(pretrain.get('gradient_steps') if isinstance(pretrain, dict) else pretrain),
                demo_buffer=demo_buffer,
            )

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
