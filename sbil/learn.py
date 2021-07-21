
import argparse
import yaml
from pathlib import Path
import importlib
import gym
from functools import partial
import os
from inspect import signature
import sys
import ast
from operator import attrgetter

import sbil
import sbil.demo
import stable_baselines3 as sb
from sbil.utils import safe_eval, make_config, make_env, make_learner, ok, EvalSaveGif, get_class
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

# for multiprocessing, the import must be in the global scope
config, gym_import = make_config()
assert {"env", "learner"} <= set(config.keys()), "env and learner are required in the yaml file."


def main():
    # unpack
    config_env = config['env']
    config_learner = config['learner']
    config_algorithm = config.get('algorithm', None)
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
                exec(code, {}, {}) # safe ?
        else:
            print("Evaluating declare")
            safe_eval(config_declare)
    m = sys.modules[__name__]

    # gym environment wrappers
    env = make_env(config_env)
    # learner
    learner = make_learner(config_learner, env, config_algorithm)
    print(
        f"Using {type(learner).__name__}"
        +(f" and {config_algorithm['algorithm']}" if config_algorithm and 'algorithm' in config_algorithm else "")
        +f" on {config_env['id']}"
    )

    if config_learn is not None and ok(config_learn):
        # callback
        if config_learn.get('callback', None) is not None:
            callback_dict = config_learn.pop('callback')
            assert 'class' in callback_dict, "You must specify class in learn->callback."
            callback_class_name = callback_dict.pop('class')
            callback_class = get_class(callback_class_name)
            if callback_class is  None:
                print("You are calling a callback that doesn't exist in learn")
                return
            if 'eval_env' in signature(callback_class).parameters:
                callback_dict['eval_env'] = env
            config_learn['callback'] = callback_class(**callback_dict)

        learner.learn(**config_learn)

    if config_save is not None and ok(config_save):
        if config_save.get('learner',None) is not None:
            model.save(config_save['learner'])
        if config_save.get('policy',None) is not None:
            model.policy.save(config_save['policy'])
        if config_save.get('env',None) is not None:# and hasattr(env, "save"):
            env.save(config_save['env'])


if __name__ == "__main__":
    main()
