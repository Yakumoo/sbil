
import argparse
import importlib

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from sbil.utils import safe_eval, make_env, make_learner, get_class

import gym
import numpy as np

config, gym_import = make_config()
assert {"env", "evaluate", "learner"} <= set(config.keys()), "env and learner are required in the yaml file."


def main():
    if importlib.util.find_spec("sb3_contrib"):
        import sb3_contrib
    else:
        sb3_contrib = None
    
    config_env = config['env']
    config_evaluate = config['evaluate']
    config_learner = config.get['learner']
    
    # gym environment
    env = make_env(config_env)
    
    # learner
    policy = config_learner['policy']
    if isinstance(policy, dict) and policy.get('load', None) is not None:
        # try to load policy without instanciating the learner
        learner_class_name = config_learner.pop('class')
        if hasattr(sb, learner_class_name):
            learner_class = getattr(sb, learner_class_name)
        elif sb3_contrib is not None:
            if hasattr(sb3_contrib, learner_class_name):
                learner_class = getattr(sb, learner_class_name)
        else:
            learner_class = get_class(learner_class_name)
            assert learner_class is not None, f"Learner class names ({learner_class_name}) are not matching"
        
        policy_bases = {
            'A2C':sb.a2c.policies.A2CPolicy,
            'DDPG':sb.ddpg.policies.DDPGPolicy,
            'DQN':sb.dqn.policies.DQNPolicy,
            'PPO':sb.ppo.policies.PPOPolicy,
            'SAC':sb.sac.policies.SACPolicy,
            'TD3':sb.td3.policies.TD3Policy,
        }
        if sb3_contrib is not None:
            policy_bases.update({
                'TQC':sb3_contrib.tqc.policies.TQCPolicy,
                'QRDQN':sb3_contrib.qrdqn.policies.QRDQNPolicy,
            })
        policy_class = get_class(policy['class'])
        if policy_class is None:
            # custom policy is not defined
            for key in policy_bases.keys():
                if issubclass(learner_class, getattr(sb, key)):
                    policy_base = policy_bases[learner_class_name]
                    policy_class = get_policy_from_name(policy['class'])
                    break
            else:
                raise Exception("Unrecognized policy")
        learner = policy_class.load(policy['load'])
    else: # load learner
        learner = make_learner(config_learner, env)
    
    render = config_evaluate.pop('render', None)
    record = config_evaluate.pop('record', None)
    
    print("mean reward={}, std reward={}".format(*evaluate_policy(learner, env, **config_evaluate)))
    
    
    

if __name__ == "__main__":
    main()
