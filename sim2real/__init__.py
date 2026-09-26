"""SO-101 sim2real reinforcement-learning pipeline.

Importing this package registers the Gymnasium environments so that
``gymnasium.make("SO101Reach-v0")`` works everywhere (training, eval, tests).
"""

from __future__ import annotations

from gymnasium.envs.registration import register

__version__ = "0.1.0"

# 在 Gymnasium 的全局注册表中建立映射
register(
    id="SO101Reach-v0",
    entry_point="sim2real.envs.so101_reach:SO101ReachEnv",
    max_episode_steps=None,  # handled inside the env (config.env.max_episode_steps)
)

register(
    id="SO101PickPlace-v0",
    entry_point="sim2real.envs.so101_pickplace:SO101PickPlaceEnv",
    max_episode_steps=None,
)

__all__ = ["__version__"]
