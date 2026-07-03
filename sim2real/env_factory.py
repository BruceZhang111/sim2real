"""Builders that turn a :class:`Config` into ready-to-train environments.

Keeps all the wrapper/vecenv/normalization wiring in one place so training,
evaluation and tests construct identical environments.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Callable

import gymnasium as gym

import sim2real  # noqa: F401  (registers SO101Reach-v0)
from sim2real.config import Config
from sim2real.wrappers import ActionLatencyWrapper, ActionNoiseWrapper


def make_single_env(
    cfg: Config,
    randomize: bool = True,
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds one wrapped SO-101 env.

    ``randomize=False`` yields a clean, nominal-dynamics env (used for eval and
    tests) so success rate reflects task skill, not luck with a soft episode.
    """

    def _init() -> gym.Env:
        dr = cfg.dr
        env_cfg = cfg.env
        if not randomize:
            dr = replace(dr, enabled=False, action_latency_steps=(0, 0), action_noise_std=0.0)
            # curriculum starts are a training aid; eval must measure the
            # from-scratch task or success rates are inflated. Zero every
            # *_init_prob field so newly added rungs can never leak into eval
            # (a forgotten one silently inflated eval success by its prob).
            zeros = {f.name: 0.0 for f in dataclasses.fields(env_cfg)
                     if f.name.endswith("_init_prob")}
            env_cfg = replace(env_cfg, **zeros)
        env = gym.make(cfg.env.env_id, config=env_cfg, dr=dr, render_mode=render_mode)
        if dr.enabled:
            if dr.action_noise_std > 0:
                env = ActionNoiseWrapper(env, dr.action_noise_std)
            lo, hi = dr.action_latency_steps
            if hi > 0:
                env = ActionLatencyWrapper(env, lo, hi)
        return env

    return _init


def make_training_venv(cfg: Config):
    """Vectorized, (optionally) normalized training environment for SB3."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    n = cfg.train.n_envs
    vec_cls = SubprocVecEnv if n > 1 else DummyVecEnv
    venv = make_vec_env(
        make_single_env(cfg, randomize=True),
        n_envs=n,
        seed=cfg.train.seed,
        vec_env_cls=vec_cls,
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    if cfg.train.normalize_obs or cfg.train.normalize_reward:
        venv = VecNormalize(
            venv,
            norm_obs=cfg.train.normalize_obs,
            norm_reward=cfg.train.normalize_reward,
            clip_obs=10.0,
            gamma=cfg.train.gamma,
        )
    return venv


def make_eval_venv(cfg: Config):
    """Single-env, nominal-dynamics eval venv (normalization added by caller)."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = make_vec_env(
        make_single_env(cfg, randomize=False),
        n_envs=1,
        seed=cfg.train.seed + 10_000,
        vec_env_cls=DummyVecEnv,
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    if cfg.train.normalize_obs:
        # Reward is never normalized at eval; obs_rms is synced from the trainer.
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    return venv
