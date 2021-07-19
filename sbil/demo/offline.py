
# BC, DRIL, ORIL
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import numpy as np
import torch as th
from torch.nn.functional import logsigmoid, mse_loss

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from sbil.demo.utils import get_demo_buffer, state_action, all_state_action

def behavioural_cloning(policy, demo_buffer, gradient_steps, learner, print_loss=None):
    is_off = isinstance(learner, OffPolicyAlgorithm)
    is_on = not is_off
    for i in range(gradient_steps):
        demo_sample = demo_buffer.sample(learner.batch_size, learner._vec_normalize_env)
        actions = policy(demo_sample.observations)
        if is_on:
            value, log_prob, entropy = policy.evaluate_actions(obs=demo_sample.observations, actions=demo_sample.actions)
            loss = -log_prob.mean()
        elif is_off and hasattr(policy, "action_dist"):
            policy(demo_sample.observations)
            loss = -policy.action_dist.log_prob(demo_sample.actions).mean()
        else:
            loss = mse_loss(input=actions, target=demo_sample.actions)
        # todo: crossentropy loss for discrete action
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()
        if print_loss is not None and i%print_loss == 0:
            print("loss:", loss.item())

def bc(
    learner: Union[OnPolicyAlgorithm, OffPolicyAlgorithm],
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    gradient_steps: int = 1000,
    print_loss: Optional[int] = None
) -> Union[OnPolicyAlgorithm, OffPolicyAlgorithm]:
    
    demo_buffer = get_demo_buffer(demo_buffer, learner)
    is_off = isinstance(learner, OffPolicyAlgorithm)
    behavioural_cloning(getattr(learner, "actor" if is_off else "policy"), demo_buffer, gradient_steps, learner)
    return learner

