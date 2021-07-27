
# RED

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import (
    DictReplayBuffer, ReplayBuffer, RolloutBuffer, DictRolloutBuffer
)
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.torch_layers import create_mlp

from sbil.demo.utils import get_demo_buffer, state_action, all_state_action
from sbil.utils import set_method, MLP, set_restore, save, get_policy, get_features_extractor
import sbil

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path
from types import MethodType
from operator import attrgetter
from functools import partial

import torch as th
from torch.nn.functional import logsigmoid, mse_loss
import numpy as np
import gym


def sample(self, batch_size, env, *args, super_, rnd, learner, state_only: bool = False, **kwargs) -> Union[DictReplayBufferSamples, ReplayBufferSamples]:
    replay_data = super_(batch_size=batch_size, env=env, *args, **kwargs)
    sa = state_action(replay_data.observations, replay_data.actions, learner, state_only)
    rewards = rnd(sa).mean(-1).view(batch_size, self.n_envs)
    replay_data.rewards[:] = th.exp(-rewards)
    return replay_data

def compute_returns_and_advantage(self, super_, rnd, learner, state_only: bool = False, *args, **kwargs) -> None:
    sa = all_state_action(self, learner, state_only)
    rewards = rnd(sa).mean(-1).view(self.buffer_size, self.n_envs)
    rewards = rewards.detach().numpy()
    self.rewards[:] = np.exp(-rewards)
    super_(*args, **kwargs)

def forward(self, data, super_=None):
    return mse_loss(input=self.mlp(data), target=self.target(data), reduction='none')


def red(
    learner: Union[OnPolicyAlgorithm, OffPolicyAlgorithm],
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    state_only: bool = False,
    net_arch: Dict[str, Union[List[int], int]] = {'target':[64, 64], 'train':[128, 64], 'latent':64},
    gradient_steps: int = 1000,
    load: Optional[str] = None,
) -> Union[OnPolicyAlgorithm, OffPolicyAlgorithm]:
    """
    Random Expert Distillation: https://arxiv.org/abs/1905.06750
    The rnd is trained before hand (here)
    The reward is modified in:
    - learner.replay_buffer._get_samples if OffPolicyAlgorithm
    - learner.rollout_buffer.compute_returns_and_advantage if OnPolicyAlgorithm

    :param learner: Stable baselines learner object
    :param demo_buffer: Demonstration replay buffer
    :param state_only: Use state only
        default is the concatenation of the state-action pair
    :param net_arch: target_net, train_net architecture and latent space dim.
    :param load: Zip file containing state_dict of the rnd to resume a training.
        It should be the same as the path used with .load(path)
    :return leaner: Decorated learner
    """
    demo_buffer = get_demo_buffer(demo_buffer, learner)
    is_off = isinstance(learner, OffPolicyAlgorithm)
    is_on = not is_off

    buffer_name = "replay_buffer" if is_off else "rollout_buffer"
    set_restore(getattr(learner, buffer_name))
    set_restore(learner, lambda self: getattr(self, buffer_name).restore())

    # rnd
    net_arch = {k.lower().strip():v for k, v in net_arch.items()}
    assert set(net_arch.keys()) == {'target', 'train', 'latent'}, (
        "target: List, train: List, latent: int are required")
    if load is None:
        learner.state_only = state_only
        learner.net_arch = net_arch

    p = get_features_extractor(learner)
    input_dim = p.features_extractor.features_dim
    if not learner.state_only:
        input_dim += get_flattened_obs_dim(learner.action_space)
    rnd = MLP(
        input_dim=input_dim,
        output_dim=learner.net_arch['latent'],
        net_arch=learner.net_arch['train'],
    )
    rnd.target = th.nn.Sequential(*create_mlp(
        input_dim=input_dim,
        output_dim=learner.net_arch['latent'],
        net_arch=learner.net_arch['train'],
        activation_fn=th.nn.LeakyReLU
    )).requires_grad_(requires_grad=False) # fixed
    rnd = rnd.to(learner.device)
    set_method(rnd, old="forward", new=forward)
    # saving and loading
    modules = {'rnd': rnd, 'rnd_optimizer': rnd.optimizer}
    if load is not None:
        load(path, modules=modules)
    set_method(learner, old="save", new=partial(save, modules=modules))

    # overwrite set_method with additional arguments
    set_method_ = partial(
        set_method,
        rnd=rnd,
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

    # train rnd
    batch_size = learner.batch_size if hasattr(learner, "batch_size") else learner.n_steps*learner.env.num_envs
    for i in range(gradient_steps):
        demo_sample = demo_buffer.sample(batch_size, env=learner._vec_normalize_env)

        sa = state_action(demo_sample.observations, demo_sample.actions, learner, state_only)
        loss = rnd(sa).mean()
        rnd.optimizer.zero_grad()
        loss.backward()
        rnd.optimizer.step()
    print("rnd loss", loss.item())

    return learner