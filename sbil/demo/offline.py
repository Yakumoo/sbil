
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
from sbil.utils import action_loss, get_policy

# TODO: scale or unscale actions
def behavioural_cloning(policy, demo_buffer, gradient_steps, batch_size, env, print_loss=None):

    for i in range(gradient_steps):
        demo_sample = demo_buffer.sample(batch_size=batch_size, env=env)
        loss = action_loss(
            policy=policy,
            observations=demo_sample.observations,
            actions=demo_sample.actions,
        ).mean()
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

    demo_buffer_ = get_demo_buffer(demo_buffer, learner)
    behavioural_cloning(
        policy=get_policy(learner),
        demo_buffer=demo_buffer_,
        gradient_steps=gradient_steps,
        batch_size=learner.batch_size if hasattr(learner, "batch_size") else learner.n_steps*learner.env.num_envs,
        env=learner._vec_normalize_env,
        print_loss=print_loss,
    )
    return learner
