
# BC, DRIL, ORIL
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import numpy as np
import torch as th
from torch.nn.functional import logsigmoid, mse_loss, cross_entropy

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from sbil.demo.utils import get_demo_buffer, state_action, all_state_action

def behavioural_cloning(policy, demo_buffer, gradient_steps, learner, print_loss=None):
    is_off = isinstance(learner, OffPolicyAlgorithm)
    is_on = not is_off

    if hasattr(policy, "actor") and policy.actor is not None:
        policy_ = policy.actor
    else:
        policy_ = policy

    for i in range(gradient_steps):
        demo_sample = demo_buffer.sample(learner.batch_size, learner._vec_normalize_env)
        if is_on: # PPO, A2C
            value, log_prob, entropy = policy_.evaluate_actions(
                obs=demo_sample.observations,
                actions=demo_sample.actions
            )
            loss = -log_prob.mean()
        elif hasattr(policy_, "action_dist"): # SAC, TQC
            policy_(demo_sample.observations)
            loss = -policy_.action_dist.log_prob(demo_sample.actions).mean()
        elif hasattr(policy_, "mu"): # TD3, DDPG
            actions = policy_(demo_sample.observations)
            loss = mse_loss(input=actions, target=demo_sample.actions)
        elif hasattr(policy_, "q_net"): # DQN
            loss = cross_entropy(
                input=policy_.q_net(demo_sample.observations),
                target=demo_sample.actions.squeeze()
            )
        elif hasattr(policy_, "quantile_net"): # QRDQN
            loss = cross_entropy(
                input=policy_.quantile_net(demo_sample.observations),
                target=demo_sample.actions.squeeze()
            )
        policy_.optimizer.zero_grad()
        loss.backward()
        policy_.optimizer.step()
        if print_loss is not None and i%print_loss == 0:
            print("loss:", loss.item())

def bc(
    learner: Union[OnPolicyAlgorithm, OffPolicyAlgorithm],
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    gradient_steps: int = 1000,
    print_loss: Optional[int] = None
) -> Union[OnPolicyAlgorithm, OffPolicyAlgorithm]:

    demo_buffer = get_demo_buffer(demo_buffer, learner)
    behavioural_cloning(learner.policy, demo_buffer, gradient_steps, learner, print_loss)
    return learner
