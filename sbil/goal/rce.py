from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
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
from sbil.utils import set_method, set_restore, train_off

class RCESamples(NamedTuple):
    observations: Union[Dict[str, th.Tensor], th.Tensor]
    actions: th.Tensor
    next_observations: Union[Dict[str, th.Tensor], th.Tensor]
    dones: th.Tensor
    n_step: th.Tensor
    futur_obs: Union[Dict[str, th.Tensor], th.Tensor]
    demo_obs: Union[Dict[str, th.Tensor], th.Tensor]
    demo_act: th.Tensor

def critic_loss(self, data, use_behaviour_policy):
    r = data['replay']
    # get actions
    if hasattr(self.actor, "action_log_prob"): # stochastic policy
        demo_act = self.actor.action_log_prob(r.demo_obs)[0] if use_behaviour_policy else r.demo_act
        next_actions, next_log_prob = self.actor.action_log_prob(r.next_observations)
        futur_actions, futur_log_prob = self.actor.action_log_prob(r.futur_obs)
    else: # determinitic policy
        def get_action(obs):
            act = self.actor_target(obs)
            noise = act.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            return (act + noise).clamp(-1, 1)
        demo_act = get_action(r.demo_obs) if use_behaviour_policy else r.demo_act
        next_actions = get_action(r.next_observations)
        futur_actions = get_action(r.futur_obs)

    # concatenate next and futur for efficency
    if isinstance(r.observations, dict):
        nf_obs = {k: th.cat((v, r.futur_obs[k])) for k,v in r.next_observations.items()}
    else:
        nf_obs = th.vstack((r.next_observations, r.futur_obs))
    nf_act = th.vstack((next_actions, futur_actions))
    nf_dones = th.cat((r.dones, r.dones))
    nf_q = self.critic_target(nf_obs, nf_act)

    if isinstance(self, (TD3, SAC)):
        with th.no_grad():
            nf_q, _ = th.min(th.sigmoid(th.cat(nf_q, dim=1)), dim=1, keepdim=True)
            nf_w = (1 - nf_dones) * nf_q
            nf_w /= (1.000001-nf_w)
            nf_γw = self.gamma**th.cat((th.ones_like(r.n_step), r.n_step)).view(-1,1) * nf_w
            # split
            next_γw, futur_γw = th.tensor_split(nf_γw, 2)
            w, futur_w = th.tensor_split(nf_w, 2)
            # The prediction is the mean
            y = (next_γw/(next_γw+1) + futur_γw/(futur_γw+1)) / 2

        value = th.cat(self.critic(r.observations, r.actions), dim=1)
        demo_value = th.cat(self.critic(r.demo_obs, demo_act), dim=1)

        # reshape to do pairwise operations (double sum)
        input = th.cat((demo_value, value), dim=0)
        target = th.cat((th.ones_like(demo_value), y.repeat(1,2)), dim=0)

    elif TQC is not None and isinstance(self, TQC):
        n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
        with th.no_grad():
            nf_q = th.sigmoid(nf_q)
            nf_q, _ = th.sort(nf_q.reshape(2*data['batch_size'], -1)) # sort
            nf_q = nf_q[:, :n_target_quantiles] # drop
            nf_w = ((1 - nf_dones) * nf_q)[:, None, None, :] # discard transitions with done
            nf_w = nf_w / (1.000001-nf_w) # classifier’s prediction (ratio) at the next time step and futur
            nf_γ = self.gamma**th.vstack((th.ones_like(r.n_step), r.n_step))
            nf_γw = nf_γ.view(2*data['batch_size'], 1, 1, 1) * nf_w
            # split
            next_γw, futur_γw = th.tensor_split(nf_γw, 2)
            w, futur_w = th.tensor_split(nf_w, 2)
            # The prediction is the mean
            y = (next_γw/(next_γw+1) + futur_γw/(futur_γw+1)) / 2

        current_quantiles = self.critic(r.observations, r.actions)
        demo_quantiles = self.critic(r.demo_obs, demo_act)

        # reshape to do pairwise operations (double sum)
        input = th.cat((demo_quantiles, current_quantiles), dim=0).unsqueeze(-1).repeat(1, 1, 1, n_target_quantiles)
        target = th.cat((th.ones_like(y), y), dim=0).repeat(1, self.critic.n_critics, self.critic.n_quantiles, 1)

    else:
        raise NotImplementedError(f"critic_loss not implemented for {self}.")

    weight = th.cat((th.ones_like(w)*(1-self.gamma), 1+self.gamma*w), dim=0)
    critic_loss = F.binary_cross_entropy_with_logits(input=input, target=target, weight=weight)

    data['critic_loss'] = critic_loss
    return {'critic_loss': critic_loss.item()}


def sample(self, batch_size, env, *args, demo_buffer, n_step, super_=None, **kwargs) -> Union[RCESamples]:
    if self.full:
        batch_inds = (np.random.randint(0, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
    else:
        batch_inds = np.random.randint(0, max(self.pos-n_step, 1), size=batch_size)

    # get next_obs
    if isinstance(self.observations, dict):
        get_obs = lambda i: self._normalize_obs({key: obs[i, 0, :] for key, obs in self.observations.items()}, env)
        is_dict = True
    else:
        get_obs = lambda i: self._normalize_obs(self.observations[i, 0, :], env)
        is_dict = False

    next_batch_inds = (batch_inds + 1) % self.buffer_size # batch indexe

    # fetch the greatest n_step value without reaching done
    n_step_, ok = np.ones(batch_size, dtype=int), np.ones(batch_size, dtype=bool)
    for i in range(1, n_step):
        ok = np.logical_and(np.logical_not(self.dones[(next_batch_inds + i) % self.buffer_size, 0]), ok)
        n_step_ += ok

    # get futur
    futur_batch_inds = (batch_inds + n_step_) % self.buffer_size

    # sample demo
    demo_sample = demo_buffer.sample(batch_size=batch_size, env=env)

    data = (
        get_obs(batch_inds),
        self.actions[batch_inds, :] if is_dict else self.actions[batch_inds, 0, :],
        get_obs(next_batch_inds),
        self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
        n_step_,
        get_obs(futur_batch_inds),
        demo_sample.observations,
        demo_sample.actions,
    )
    def to_torch(x):
        if isinstance(x, dict):
            return {k:self.to_torch(v) for k,v in x.items()}
        return self.to_torch(x)

    return RCESamples(*tuple(map(to_torch, data)))

def rce(
    learner: OffPolicyAlgorithm,
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    n_step=10,
    use_behaviour_policy: bool = False,
) -> OffPolicyAlgorithm:
    """
    Recursive classification of examples: https://arxiv.org/abs/2103.12656
    We suppose observations are stored sequentially: o_0, o_1, o_2, ...
    so we can access to futur observations for n_step

    :param learner: Stable baselines learner object
    :param demo_buffer: Demonstration replay buffer containing success examples (not trajectories).
    :param n_step: Number of futur step for the n_step return.
    :param use_behaviour_policy: Choose to use the current policy to estimate the action
        of the demonstration. Defaut to False.
    :return learner: Decorated learner
    """
    is_dqn = hasattr(learner, "q_net") or hasattr(learner, "quantile_net")
    assert not is_dqn, "DQN like learner is not supported."

    demo_buffer_ = get_demo_buffer(demo_buffer, learner)
    # assert demo_buffer_.optimize_memory_usage, "optimize_memory_usage must be set to True in order to use n_step."
    set_restore(learner)


    set_method(
        learner,
        old="train",
        new=train_off,
        critic_loss=partial(critic_loss, use_behaviour_policy=use_behaviour_policy),
    )

    set_method(
        learner.replay_buffer,
        old="sample",
        new=sample,
        demo_buffer=demo_buffer_,
        n_step=n_step,
    )

    return learner
