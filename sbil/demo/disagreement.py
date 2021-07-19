from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from pathlib import Path
from functools import partial

import numpy as np
import torch as th
from torch.nn.functional import logsigmoid, mse_loss

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer, RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from sbil.demo.offline import behavioural_cloning
from sbil.demo.utils import get_demo_buffer, state_action, all_state_action
from sbil.utils import set_method, MLP, set_restore, save

def train_off(self, gradient_steps, super_, demo_buffer, *args, **kwargs) -> None:
    behavioural_cloning(self.actor, demo_buffer, gradient_steps, self)
    super_(gradient_steps, *args, **kwargs)

def train_on(self, super_, demo_buffer, *args, **kwargs) -> None:
    behavioural_cloning(self.policy, demo_buffer, self.n_epochs, self)
    super_(*args, **kwargs)
        
def sample(self, batch_size, env, *args, super_, Π, **kwargs) -> Union[DictReplayBufferSamples, ReplayBufferSamples]:
    replay_data = super_(batch_size=batch_size, env=env, *args, **kwargs)
    if hasattr(Π[0], "action_dist"):
        eval = [None]*len(Π)
        for i,π in enumerate(Π):
            π(replay_data.observations)
            eval[i] = π.action_dist.log_prob(replay_data.actions)
        # variance of log_prob
        var = th.var(th.stack(eval, dim=0), unbiased=True, dim=0)
        rewards = - var
    else:
        eval = [π(self.to_torch(self.observations)) for π in Π]
        # variance of actions, because log_prob can not be computed like DQN, TD3
        # mean over action space
        # just negative variance as in the paper
        rewards = -th.var(th.tensor(eval), unbiased=True, dim=0).mean(-1)
    # todo: discrete actions ?
        
    replay_data.rewards[:] = rewards.view(batch_size, self.n_envs)
    
    return replay_data

def compute_returns_and_advantage(self, super_, Π, *args, **kwargs) -> None:
    obs = self.to_torch(self.observations).view(self.buffer_size*self.n_envs, -1)
    actions = self.to_torch(self.actions).view(self.buffer_size*self.n_envs, -1)
    log_prob = [π.evaluate_actions(obs, actions)[1] for π in Π]
    log_prob = th.stack(log_prob, dim=0)
    # variance of log_prob not between 0 and 1
    var = th.var(log_prob, unbiased=True, dim=0)
    self.rewards[:] = -var.view(self.buffer_size, self.n_envs).detach().numpy()
    super_(*args, **kwargs)

def dril(
    learner: Union[OnPolicyAlgorithm, OffPolicyAlgorithm],
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    gradient_steps: int = 1,
    E: int = 2,
    #q: float = 0.98
    print_loss: Optional[int] = None,
) -> Union[OnPolicyAlgorithm, OffPolicyAlgorithm]:
    """
    Disagreement-Regularized Imitation Learning: https://openreview.net/forum?id=rkgbYyHtwB
    todo: dropout
    The clip trick is not used to eliminate the hyperparameter q (quantile cutoff).
    They said: "costs are always positive (which corresponds to a reward which is always negative),
    the agent may learn to terminate the episode early in order to minimize the total cost incurred"
    Thus, use AbsorbingState wrapper to learn whether early stopping is a desired behaviour.
    The variance of actions is used instead of variance of the probability of taking the action a
    when log_prob is not available like DQN, TD3.
    Set gradient_steps to 1 to match the paper: bc and variance loss at each gradient step.
    
    :param E: Number of policies in ensemble
    #:param q: quantile cutoff
    """
    demo_buffer = get_demo_buffer(demo_buffer, learner)
    
    is_off = isinstance(learner, OffPolicyAlgorithm)
    is_on = not is_off
    buffer_name = "replay_buffer" if is_off else "rollout_buffer"
    set_restore(getattr(learner, buffer_name))
    set_restore(learner, lambda self: getattr(self, buffer_name).restore())
    p = getattr(learner, "actor" if is_off else "policy")
    

    π = getattr(learner, "actor" if is_off else "policy")
    Π = [deepcopy(π) for i in range(E)]
    rng = np.random.default_rng()
    s = demo_buffer.size()
    # train ensemble,
    # can not parallelize because the env is local
    for π_ in Π:
        d = deepcopy(demo_buffer)
        indexes = rng.choice(s, s, replace=True)
        # rewards, dones and next_observations are useless
        if isinstance(d, DictReplayBuffer):
            d.observations = {key: obs[indexes] for (key, obs) in d.observations.items()}
        elif isinstance(d, ReplayBuffer):
            d.observations = d.observations[indexes]
        d.actions = d.actions[indexes]
        behavioural_cloning(π_, d, gradient_steps, learner)
    
    # modify train()
    set_method(
        learner,
        old="train",
        new=train_off if is_off else train_on,
        demo_buffer=demo_buffer,
    )
    # overwrite set_method with additional arguments
    set_method_ = partial(
        set_method,
        Π=Π,
    )

    # modify the buffer to change the reward
    if is_off:
        set_method_(learner.replay_buffer, old="sample", new=sample)
    elif is_on:
        set_method_(learner.rollout_buffer, old="compute_returns_and_advantage", new=compute_returns_and_advantage)
    else:
        raise NotImplementedError()
    
    return learner
