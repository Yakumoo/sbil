from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from pathlib import Path
from functools import partial

import numpy as np
import torch as th
from torch.nn.functional import logsigmoid, mse_loss, softmax

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
from sbil.utils import set_method, set_restore, get_policy, action_loss, train_off


def actor_loss(self, data, Π, demo_buffer):
    # the critic is not needed.
    r = data['replay']
    demo_sample = demo_buffer.sample(batch_size=self.batch_size, env=self._vec_normalize_env)
    loss = action_loss(get_policy(self), demo_sample.observations, demo_sample.actions) # BC loss
    action = self.policy._predict(r.observations, deterministic=True)
    loss += th.var(th.stack(tuple(action_loss(π, r.observations, action) for π in Π), dim=0), unbiased=True, dim=0) # Variance
    loss = loss.mean()
    data['actor_loss'] = loss
    return {'actor_loss': loss.item()}

def dril(
    learner: OffPolicyAlgorithm,
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    gradient_steps: int = 1000,
    n_policies: int = 3,
    #q: float = 0.98
    print_loss: Optional[int] = None,
) -> OffPolicyAlgorithm:
    """
    Disagreement-Regularized Imitation Learning: https://openreview.net/forum?id=rkgbYyHtwB
    todo: dropout
    The clip trick is not used to eliminate the hyperparameter q (quantile cutoff).
    They said: "costs are always positive (which corresponds to a reward which
    is always negative), the agent may learn to terminate the episode early in
    order to minimize the total cost incurred". Thus, use AbsorbingState wrapper
    to learn whether early stopping is a desired behaviour.

    :param n_policies: Number of policies in ensemble
    #:param q: quantile cutoff

    :return learner: Decorated learner
    """
    demo_buffer_ = get_demo_buffer(demo_buffer, learner)

    is_off = isinstance(learner, OffPolicyAlgorithm)
    is_on = not is_off
    buffer_name = "replay_buffer" if is_off else "rollout_buffer"
    set_restore(getattr(learner, buffer_name))
    set_restore(learner, lambda self: getattr(self, buffer_name).restore())
    batch_size = learner.batch_size if hasattr(learner, "batch_size") else learner.n_steps*learner.env.num_envs

    π = get_policy(learner)
    Π = [deepcopy(π) for i in range(n_policies)]
    rng = np.random.default_rng()
    s = demo_buffer_.size()
    # train ensemble,
    # can not parallelize because the env is local
    for π_ in Π:
        for layer in π_.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        d = deepcopy(demo_buffer_)
        indexes = rng.choice(s, s, replace=True)
        # rewards, dones and next_observations are useless
        if isinstance(d, DictReplayBuffer):
            d.observations = {key: obs[indexes] for (key, obs) in d.observations.items()}
        elif isinstance(d, ReplayBuffer):
            d.observations = d.observations[indexes]
        d.actions = d.actions[indexes]
        behavioural_cloning(π_, d, gradient_steps, batch_size, learner._vec_normalize_env)
        π_ = π_.requires_grad_(requires_grad=False) # fix

    # modify train()
    set_method(
        learner,
        old="train",
        new=train_off,
        actor_loss=partial(actor_loss, Π=Π, demo_buffer=demo_buffer),
    )

    return learner
