# GAIL, DAC, PURL
# Off-PAC sasaki 2019
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer, RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from sbil.demo.utils import get_demo_buffer, state_action, all_state_action
from sbil.utils import set_method, MLP, set_restore, save, get_policy, get_features_extractor
import sbil

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path
from types import MethodType
from operator import attrgetter
from functools import partial

import torch as th
from torch.nn.functional import logsigmoid, mse_loss, binary_cross_entropy_with_logits
import numpy as np
import gym



def discriminator_step(discriminator, buffer_sample, demo_buffer, learner, state_only, λ=1, center=1, η=None):
    batch_size = buffer_sample.actions.shape[0]
    extract_features = get_policy(learner).extract_features
    demo_sample = demo_buffer.sample(batch_size, learner._vec_normalize_env)
    # get state-action pairs
    sa = state_action(buffer_sample.observations, buffer_sample.actions, learner, state_only)
    demo_sa = state_action(demo_sample.observations, demo_sample.actions, learner, state_only)

    # regularization: zero-centered gradient penalty: https://openreview.net/pdf?id=ByxPYjC5KQ
    α = th.rand(batch_size,1).to(learner.device)
    inputs = th.autograd.Variable(sa*α + (1-α)*demo_sa, requires_grad=True).to(learner.device)
    outputs = discriminator(inputs)
    gradients = th.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=th.ones(outputs.size()).to(learner.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    penalty = λ * th.square(th.linalg.norm(gradients)-center)
    # discrimination loss
    if η is None: # positive negative
        input = discriminator(th.cat((demo_sa, sa), dim=0)).squeeze()
        target = th.cat((th.ones(batch_size), th.zeros(batch_size)), dim=0)
        weight = None
    else: # Positive unlabeled
        input = discriminator(th.cat((demo_sa, sa, demo_sa), dim=0)).squeeze()
        target = th.cat((th.ones(batch_size), th.zeros(batch_size*2)), dim=0)
        weight = th.cat((th.ones(batch_size)*η, th.ones(batch_size), -th.ones(batch_size)*η), dim=0).to(learner.device)
    loss = binary_cross_entropy_with_logits(input=input, target=target.to(learner.device), weight=weight)

    loss = loss.mean() + penalty.mean()
    discriminator.optimizer.zero_grad()
    loss.backward()
    discriminator.optimizer.step()

    return loss.item()


def train_off(self, gradient_steps, super_, discriminator, demo_buffer, state_only: bool = False, η=None, *args, **kwargs) -> None:
    for gradient_step in range(gradient_steps):
        replay_data = self.replay_buffer.sample(self.batch_size, self._vec_normalize_env)
        discriminator_step(discriminator, replay_data, demo_buffer, self, state_only, η=η)
    super_(gradient_steps, *args, **kwargs)

def train_on(self, super_, discriminator, demo_buffer, state_only: bool = False, η=None, *args, **kwargs) -> None:
    for epoch in range(getattr(self, "n_epochs", 1)):
        for rollout_data in self.rollout_buffer.get(batch_size=getattr(self, "batch_size", None)):
            discriminator_step(discriminator, rollout_data, demo_buffer, self, state_only, η=η)
    super_(*args, **kwargs)

def sample(self, batch_size, env, *args, super_, discriminator, learner, state_only: bool = False, **kwargs) -> Union[DictReplayBufferSamples, ReplayBufferSamples]:
    replay_data = super_(batch_size=batch_size, env=env, *args, **kwargs)
    sa = state_action(replay_data.observations, replay_data.actions, learner, state_only)
    with th.no_grad():
        replay_data.rewards[:] = discriminator(sa)

    return replay_data

def compute_returns_and_advantage(self, super_, discriminator, learner, state_only: bool = False, *args, **kwargs) -> None:
    sa = all_state_action(self, learner, state_only)
    with th.no_grad():
        out = discriminator(sa)
    self.rewards[:] = out.view(self.buffer_size, self.n_envs).cpu().numpy() # unstack
    super_(*args, **kwargs)

def adversarial(
    learner: Union[OnPolicyAlgorithm, OffPolicyAlgorithm],
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    state_only: bool = False,
    net_arch: List[int] = [64, 64],
    η: Optional[float] = None,
    load: Optional[str] = None,
) -> Union[OnPolicyAlgorithm, OffPolicyAlgorithm]:
    """
    Adversarial imitation learning decorator: https://arxiv.org/pdf/1809.02925.pdf
    The discriminator is trained in learner.train
    The reward is modified in:
    - learner.replay_buffer._get_samples if OffPolicyAlgorithm
    - learner.rollout_buffer.compute_returns_and_advantage if OnPolicyAlgorithm

    :param learner: Stable baselines learner object
    :param demo_buffer: Demonstration replay buffer
    :param state_only: Use state only
        default is the concatenation of the state-action pair
    :param η: Positive class prior, usually set to 0.5 in papers. PURL is disabled by default.
        https://arxiv.org/abs/1911.00459
    :param load: Zip file containing state_dict of the discriminator to resume a training.
        It should be the same as the path used with .load(path)
    :return leaner: Decorated learner
    """
    demo_buffer = get_demo_buffer(demo_buffer, learner)
    is_off = isinstance(learner, OffPolicyAlgorithm)
    is_on = not is_off

    buffer_name = "replay_buffer" if is_off else "rollout_buffer"
    set_restore(getattr(learner, buffer_name))
    set_restore(learner, lambda self: getattr(self, buffer_name).restore())


    # discriminator
    if load is None:
        learner.state_only = state_only
        learner.net_arch = net_arch
    p = get_features_extractor(learner)
    input_dim = p.features_extractor.features_dim
    if not learner.state_only:
        input_dim += get_flattened_obs_dim(learner.action_space)
    discriminator = MLP(
        input_dim=input_dim,
        output_dim=1,
        net_arch=learner.net_arch,
        spectral_norm=True,
    ).to(learner.device)
    modules = {'discriminator': discriminator, 'discriminator_optimizer': discriminator.optimizer}
    if load is not None:
        load(path, modules=modules)
    set_method(learner, old="save", new=partial(save, modules=modules))

    # modify train() to train the discriminator
    set_method(
        learner,
        old="train",
        new=train_off if is_off else train_on,
        discriminator=discriminator,
        demo_buffer=demo_buffer,
        state_only=state_only,
        η=η,
    )

    # overwrite set_method with additional arguments
    set_method_ = partial(
        set_method,
        discriminator=discriminator,
        state_only=state_only,
        learner=learner,
    )

    # modify the buffer to change the reward
    if is_off:
        set_method_(learner.replay_buffer, old="sample", new=sample)
    elif is_on:
        set_method_(learner.rollout_buffer, old="compute_returns_and_advantage", new=compute_returns_and_advantage)
    else:
        raise NotImplementedError()
    return learner
