"""SimBa actor/critic backbones for Stable-Baselines3 PPO.

The observation running statistics stay in the environment-side
``VecNormalize`` wrapper. This module implements the network that follows
those normalized observations:

    Linear -> repeated pre-LayerNorm residual blocks -> LayerNorm

Each residual block uses the paper's fixed inverted bottleneck
``d_h -> 4 * d_h -> d_h`` with a ReLU between the two linear layers.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device


SIMBA_EXPANSION_FACTOR = 4
SIMBA_LAYER_NORM_EPS = 1e-6


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class SimBaBlock(nn.Module):
    """Pre-LayerNorm residual MLP block from the SimBa paper."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        _validate_positive_int("hidden_dim", hidden_dim)
        bottleneck_dim = SIMBA_EXPANSION_FACTOR * hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=SIMBA_LAYER_NORM_EPS)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        return features + self.mlp(self.layer_norm(features))


def _build_simba_network(feature_dim: int, hidden_dim: int, n_blocks: int) -> nn.Sequential:
    _validate_positive_int("feature_dim", feature_dim)
    _validate_positive_int("hidden_dim", hidden_dim)
    _validate_positive_int("n_blocks", n_blocks)
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        *(SimBaBlock(hidden_dim) for _ in range(n_blocks)),
        nn.LayerNorm(hidden_dim, eps=SIMBA_LAYER_NORM_EPS),
    )


class SimBaMlpExtractor(nn.Module):
    """Independent SimBa backbones with the SB3 ``MlpExtractor`` interface."""

    def __init__(
        self,
        feature_dim: int,
        actor_hidden_dim: int = 256,
        actor_n_blocks: int = 1,
        critic_hidden_dim: int = 512,
        critic_n_blocks: int = 2,
        device: th.device | str = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        self.latent_dim_pi = actor_hidden_dim
        self.latent_dim_vf = critic_hidden_dim
        self.policy_net = _build_simba_network(
            feature_dim, actor_hidden_dim, actor_n_blocks
        ).to(device)
        self.value_net = _build_simba_network(
            feature_dim, critic_hidden_dim, critic_n_blocks
        ).to(device)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class SimBaActorCriticPolicy(ActorCriticPolicy):
    """SB3 actor-critic policy that swaps only the latent MLP extractor."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args: Any,
        simba_actor_hidden_dim: int = 256,
        simba_actor_n_blocks: int = 1,
        simba_critic_hidden_dim: int = 512,
        simba_critic_n_blocks: int = 2,
        **kwargs: Any,
    ) -> None:
        # ActorCriticPolicy.__init__ calls the overridden
        # _build_mlp_extractor(), so these attributes must exist first.
        for name, value in (
            ("simba_actor_hidden_dim", simba_actor_hidden_dim),
            ("simba_actor_n_blocks", simba_actor_n_blocks),
            ("simba_critic_hidden_dim", simba_critic_hidden_dim),
            ("simba_critic_n_blocks", simba_critic_n_blocks),
        ):
            _validate_positive_int(name, value)
            setattr(self, name, value)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SimBaMlpExtractor(
            feature_dim=self.features_dim,
            actor_hidden_dim=self.simba_actor_hidden_dim,
            actor_n_blocks=self.simba_actor_n_blocks,
            critic_hidden_dim=self.simba_critic_hidden_dim,
            critic_n_blocks=self.simba_critic_n_blocks,
            device=self.device,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            simba_actor_hidden_dim=self.simba_actor_hidden_dim,
            simba_actor_n_blocks=self.simba_actor_n_blocks,
            simba_critic_hidden_dim=self.simba_critic_hidden_dim,
            simba_critic_n_blocks=self.simba_critic_n_blocks,
        )
        return data
