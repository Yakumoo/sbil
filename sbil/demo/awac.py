
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from pathlib import Path
from functools import partial

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import SAC, TD3, DQN
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer, RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
try:
    from sb3_contrib import TQC, QRDQN
except ImportError:
    TQC, QRDN = None, None

from sbil.demo.buffer import double_buffer
from sbil.demo.utils import get_demo_buffer, state_action, all_state_action
from sbil.utils import set_method, MLP, set_restore, save, get_policy, action_loss, train_off, actor_loss, optimize_actor, optimize_critic


def actor_loss_(self, data, λ, mix):
    obs = data['replay'].observations
    act = data['replay'].actions
    is_dqn = hasattr(self, "q_net") or hasattr(self, "quantile_net")
    key = "actor_loss"

    if isinstance(self, TD3):
        target = self.critic.q1_forward(obs, act)
        current = self.critic.q1_forward(obs, self.actor(obs))
    elif isinstance(self, SAC):
        target, _ = th.min(th.cat(self.critic.forward(obs, act), dim=1), dim=1, keepdim=True)
        current, _ = th.min(th.cat(self.critic.forward(obs, data['actions_pi']), dim=1), dim=1, keepdim=True)
    elif is_dqn:
        target = data['current_q_values']
        logits = data['current_q_values_all']
        if QRDQN is not None and isinstance(self, QRDQN):
            target = target.mean(dim=1, keepdim=True)
            logits = logits.mean(dim=1)
        prob = F.softmax(logits, dim=-1)
        current = (prob * logits).sum(dim=-1, keepdims=True)
    elif TQC is not None and isinstance(self, TQC):
        target = self.critic(obs, act).mean(dim=2).mean(dim=1, keepdim=True)
        current = self.critic(obs, data["actions_pi"]).mean(dim=2).mean(dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"Actor_loss_ not implemented for {self}.")

    advantage = target - current
    loss = action_loss(get_policy(self), obs, act)
    if mix:
        α = (advantage > 0) if λ is None else F.sigmoid(advantage/λ)
        loss = α*loss - (1-α)*current # mix both loss, minus sign is critic maximization
    else: # regression loss only
        loss *= (advantage > 0) if λ is None else F.softplus(advantage, beta=1/λ) # weight or mask
    if 'ent_coef' in data:
        loss += data['ent_coef'] * data['log_prob']
    loss = loss.mean()

    if is_dqn:
        key = "loss"
        loss += data['loss'] # critic + regression loss

    data[key] = loss
    return {key: loss.item()}

def awac(
    learner: OffPolicyAlgorithm,
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    λ = 0.1,
    mix: bool = False,
    demo_rate: Optional[float] = None,
) -> OffPolicyAlgorithm:
    """
    Advantage weighted actor critic: https://arxiv.org/abs/2006.09359v6
    The actor loss is changed in actor_loss and the learner loads the replay buffer with load_replay_buffer.
    Softplus is used instead of exponential to avoid exploding weight.

    :param λ: Loss scaler. If set to None, hard weight (0 or 1) is used
    :param mix: Mix regression loss and critic maximization loss
    :param demo_rate: Proportion of demo samples during sampling
        Useful if you gave an expert policy in the demo_buffer.
        If you provide a random policy for exploration, consider not using it.
    """
    is_dqn = hasattr(learner, "q_net") or hasattr(learner, "quantile_net")
    assert not (is_dqn and mix), "You can not use mix and a DQN like learner."

    if demo_rate is not None:
        learner = double_buffer(learner, demo_buffer=demo_buffer, demo_rate=demo_rate)
    elif isinstance(demo_buffer, str):
        learner.load_replay_buffer(demo_buffer)
    else:
        demo_buffer_ = get_demo_buffer(demo_buffer, learner)
        learner.replay_buffer = demo_buffer_
        learner.replay_buffer.device = learner.device
    set_restore(learner)


    set_method(
        learner,
        old="train",
        new=train_off,
        actor_loss=partial(actor_loss_, λ=λ, mix=mix),
        # switch optimize_critic and optimize_actor if is_dqn
        optimize_actor=optimize_critic if is_dqn else optimize_actor,
        optimize_critic=(lambda self, data: None) if is_dqn else optimize_critic,
    )
    return learner
