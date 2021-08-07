import numpy as np
import pytest
import gym

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import TQC, QRDQN
from sbil.demo import adversarial, double_buffer, awac, dril, pwil, bc, red, gmmil
from sbil.data.generate_demo import generate_demo

def run(env_id, learner_class, algorithm):
    env = gym.make(env_id)
    learner = learner_class(env=env, policy="MlpPolicy")
    learner = algorithm(learner, demo_buffer=generate_demo(env, buffer_size=100))
    learner.learn(500)

@pytest.mark.parametrize("learner_class", [TD3, SAC, TQC])
@pytest.mark.parametrize("algorithm", [adversarial, pwil, awac, dril, bc, red, double_buffer])
def test_off(learner_class, algorithm):
    run('MountainCarContinuous-v0', learner_class, algorithm)

@pytest.mark.parametrize("env_id", ['MountainCarContinuous-v0', 'CartPole-v0'])
@pytest.mark.parametrize("learner_class", [PPO, A2C])
@pytest.mark.parametrize("algorithm", [adversarial, bc, red, gmmil])
def test_on(env_id, learner_class, algorithm):
    run(env_id, learner_class, algorithm)

@pytest.mark.parametrize("learner_class", [DQN, QRDQN])
@pytest.mark.parametrize("algorithm", [adversarial, pwil, awac, dril, bc, red, double_buffer])
def test_dqn(learner_class, algorithm):
    run("CartPole-v0", learner_class, algorithm)
