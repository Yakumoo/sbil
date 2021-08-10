import numpy as np
import pytest
import gym
from gym import spaces
from gym.envs.registration import EnvSpec

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.envs import FakeImageEnv, SimpleMultiObsEnv
from stable_baselines3.common.preprocessing import is_image_space

from sb3_contrib import TQC, QRDQN
from sbil.demo import adversarial, double_buffer, awac, dril, pwil, bc, red, gmmil
from sbil.data.generate_demo import generate_demo

def run(env, learner_class, algorithm, demo_policy=None):

    if isinstance(env, str):
        env = gym.make(env)
    if getattr(env, "spec", None) is None:
        env.spec = EnvSpec("dummy-v0")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    if env.observation_space.shape is None:
        policy = "MultiInputPolicy"
    elif is_image_space(env.observation_space):
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"

    learner = learner_class(env=env, policy=policy)
    learner = algorithm(learner, demo_buffer=generate_demo(env, buffer_size=100, policy=demo_policy))
    learner.learn(300)

@pytest.mark.parametrize("learner_class", [TD3, SAC, TQC])
@pytest.mark.parametrize("algorithm", [adversarial, awac, bc, double_buffer, dril, pwil, red])
def test_off(learner_class, algorithm):
    run('MountainCarContinuous-v0', learner_class, algorithm)
    run(FakeImageEnv(discrete=False), learner_class, algorithm, demo_policy="random")
    run(SimpleMultiObsEnv(discrete_actions=False), learner_class, algorithm, demo_policy="random")

@pytest.mark.parametrize("learner_class", [PPO, A2C])
@pytest.mark.parametrize("algorithm", [adversarial, bc, red, gmmil])
def test_on(learner_class, algorithm):
    run('MountainCarContinuous-v0', learner_class, algorithm)
    run('CartPole-v0', learner_class, algorithm)
    run(FakeImageEnv(discrete=False), learner_class, algorithm, demo_policy="random")
    run(SimpleMultiObsEnv(discrete_actions=False), learner_class, algorithm, demo_policy="random")

@pytest.mark.parametrize("learner_class", [DQN, QRDQN])
@pytest.mark.parametrize("algorithm", [adversarial, pwil, awac, dril, bc, red, double_buffer])
def test_dqn(learner_class, algorithm):
    """ Offpolicy discrete actions """
    run("CartPole-v0", learner_class, algorithm)
    run(FakeImageEnv(), learner_class, algorithm, demo_policy="random")
    run(SimpleMultiObsEnv(), learner_class, algorithm, demo_policy="random")
