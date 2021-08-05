
# PWIL
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import (
    DictReplayBuffer, ReplayBuffer, RolloutBuffer, DictRolloutBuffer
)
from stable_baselines3.her.her_replay_buffer import get_time_limit
from stable_baselines3.common.preprocessing import get_action_dim, preprocess_obs

from sbil.demo.utils import get_demo_buffer, state_action, all_state_action
from sbil.utils import set_method, MLP, set_restore, save, _excluded_save_params, get_features_extractor
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
from sklearn.preprocessing import StandardScaler

def _store_transition(
    self,
    replay_buffer: ReplayBuffer,
    buffer_action: np.ndarray,
    new_obs: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    infos: List[Dict[str, Any]],
    super_,
    state_only,
    demo_sa,
    T,
    scaler,
) -> None:
    """
    Store transition in the replay buffer.
    We store the normalized action and the unnormalized observation.
    It also handles terminal observations (because VecEnv resets automatically).
    :param replay_buffer: Replay buffer object where to store the transition.
    :param buffer_action: normalized action
    :param new_obs: next observation in the current episode
        or first observation of the episode (when done is True)
    :param reward: reward for the current transition
    :param done: Termination signal
    :param infos: List of additional information about the transition.
        It may contain the terminal observations and information about timeout.
    """
    if self.n > T: # overflow
        self.n += 1
        if self.n == T:
            print("Episode time step more than expected.")
        if done: # re-initialize back to the original
            print("Episode done with overflow:", self.n)
            self.pool_sa = np.array(demo_sa)
            s = len(self.pool_sa)
            self.pool_w = np.ones(s) / s + 1e-6
            self.n = 0
        return
    # Store only the unnormalized version
    if self._vec_normalize_env is not None:
        new_obs_ = self._vec_normalize_env.get_original_obs()
        reward_ = self._vec_normalize_env.get_original_reward()
    else:
        # Avoid changing the original ones
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

    # get state-action pair
    t = lambda x: self.replay_buffer.to_torch(x)
    obs = self._last_original_obs
    obs = {k: t(v) for k,v in obs.items()} if isinstance(obs, dict) else t(obs)
    sa = state_action(obs, t(buffer_action), self, state_only).detach().numpy()
    sa = scaler.transform(sa)[0] # standardized state-action

    # upper bound wasserstein distance greedy coupling
    cost = 0
    weight = 1/T - 1e-6
    norm = np.linalg.norm(self.pool_sa-sa, axis=-1)
    argsort = np.argsort(norm)
    i = 0
    while weight > 0:
        j = argsort[i]
        demo_weight = self.pool_w[j]
        demo_dist = norm[j]
        if weight >= demo_weight:
            cost += demo_weight * demo_dist
            weight -= demo_weight
            i += 1
        else:
            cost += weight * demo_dist
            self.pool_w[j] -= weight
            weight = 0


    # delete visited
    to_delete = argsort[np.arange(i)]
    self.pool_sa = np.delete(self.pool_sa, to_delete, axis=0)
    self.pool_w = np.delete(self.pool_w, to_delete, axis=0)

    reward_ = self.α * np.exp(-self.σ * cost)
    self.n += 1#; print(self.n, len(self.pool_w), i)
    if done: # re-initialize back to the original
        self.pool_sa = np.array(demo_sa)
        s = len(self.pool_sa)
        self.pool_w = np.ones(s) / s + 1e-6
        self.n = 0

    # As the VecEnv resets automatically, new_obs is already the
    # first observation of the next episode
    if done and infos[0].get("terminal_observation") is not None:
        next_obs = infos[0]["terminal_observation"]
        # VecNormalize normalizes the terminal observation
        if self._vec_normalize_env is not None:
            next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
    else:
        next_obs = new_obs_

    replay_buffer.add(
        self._last_original_obs,
        next_obs,
        buffer_action,
        reward_,
        done,
        infos,
    )

    self._last_obs = new_obs
    # Save the unnormalized observation
    if self._vec_normalize_env is not None:
        self._last_original_obs = new_obs_


def pwil(
    learner: OffPolicyAlgorithm,
    demo_buffer: Union[DictReplayBuffer, ReplayBuffer, str, Path],
    state_only: bool = False,
    α: float = 5,
    β: float = 5,
) -> OffPolicyAlgorithm:
    """
    Primal Wasserstein Imitation Learning:
    https://ai.googleblog.com/2020/09/imitation-learning-in-low-data-regime.html
    https://arxiv.org/pdf/2006.04678.pdf
    https://github.com/google-research/google-research/tree/master/pwil
    The reward is modified in _store_transition
    If demo_buffer.size() is big, the algorithm slows down. The reward is
    computed at each time step with a argsort on the demo buffer.
    It can not be parallalized easily.

    :param learner: Stable baselines learner object
    :param demo_buffer: Demonstration replay buffer
    :param state_only: Use state only
        default is the concatenation of the state-action pair
    :param α,β: reward scaler hyperparameters
    :return leaner: Decorated learner
    """
    assert learner.env.num_envs == 1, "pwil only support a single environment."
    T = get_time_limit(learner.env, None) # get time limit or raise

    set_restore(learner)
    demo_buffer = get_demo_buffer(demo_buffer, learner)
    p = get_features_extractor(learner)

    # get state-action pair from the demo_buffer
    s = demo_buffer.size()
    t = lambda x: demo_buffer.to_torch(x).view(s*demo_buffer.n_envs, -1)
    demo_obs = demo_buffer.observations
    if isinstance(demo_obs, dict):
        demo_obs = {k:t(v[:s]) for k, v in demo_obs.items()}
    else:
        demo_obs = t(demo_obs[:s])
    demo_sa = state_action(demo_obs, t(demo_buffer.actions[:s]), learner, state_only).numpy()

    scaler = StandardScaler()
    demo_sa = scaler.fit_transform(demo_sa) # stadardize

    learner.pool_sa = np.array(demo_sa) # demo pool state-action
    learner.pool_w = np.ones(s) / s + 1e-6 # demo pool weights

    # precompute reward scale
    learner.α = α
    σ = p.features_extractor.features_dim
    if not state_only:
        σ += get_action_dim(learner.action_space)
    σ = β * T / np.sqrt(σ)
    learner.σ = σ.item()
    learner.n = 0

    set_method(
        learner,
        old="_store_transition",
        new=_store_transition,
        state_only=state_only,
        demo_sa=demo_sa,
        T=T,
        scaler=scaler,
    )
    set_method(
        learner,
        old="_excluded_save_params",
        new=_excluded_save_params,
        additionals=["pool_sa", "pool_w", "n"] # They are useless
    )

    return learner
