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
# 自动执行sim2real/__init__.py
from sim2real.config import Config
from sim2real.wrappers import ActionLatencyWrapper, ActionNoiseWrapper


# 接收配置，返回一个“以后调用时才真正创建环境”的无参数函数
# builder = make_single_env(cfg)  # 此时还没有创建 MuJoCo 环境
# env = builder()                 # 调用 _init，此时才真正创建环境
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
        # eval模式下取消部分扰动
        if not randomize:
            # 基于原对象创建一个新对象，并替换指定字段
            dr = replace(dr, enabled=False, action_latency_steps=(0, 0), action_noise_std=0.0)
            # curriculum starts are a training aid; eval must measure the
            # from-scratch task or success rates are inflated. Zero every
            # *_init_prob field so newly added rungs can never leak into eval
            # (a forgotten one silently inflated eval success by its prob).
            zeros = {f.name: 0.0 for f in dataclasses.fields(env_cfg)
                     if f.name.endswith("_init_prob")}
            env_cfg = replace(env_cfg, **zeros)
        # 构建reach/pickplace环境
        env = gym.make(cfg.env.env_id, config=env_cfg, dr=dr, render_mode=render_mode)
        # ActionLatencyWrapper.step(action)
        #     │
        #     ├─ 将新动作放入延迟队列
        #     ├─ 从队列取出之前的动作
        #     ▼
        # ActionNoiseWrapper.step(delayed_action)
        #     │
        #     ├─ 添加高斯噪声
        #     ▼
        # SO101ReachEnv.step(noisy_delayed_action)
        if dr.enabled:
            if dr.action_noise_std > 0:
                env = ActionNoiseWrapper(env, dr.action_noise_std)
            lo, hi = dr.action_latency_steps
            if hi > 0:
                env = ActionLatencyWrapper(env, lo, hi)
        return env

    return _init


"""make_training_venv() 使用 make_single_env() 提供的环境构造器创建一个或多个训练环境，
通过 DummyVecEnv 或 SubprocVecEnv 统一成 SB3 的批量接口，添加成功率监控，并根据配置在最外层增加观测和奖励归一化"""
def make_training_venv(cfg: Config):
    """Vectorized, (optionally) normalized training environment for SB3."""
    # 向量环境同时管理多个环境
    from stable_baselines3.common.env_util import make_vec_env
    # make_vec_env 负责批量创建环境，并为每个环境添加 Monitor 等标准组件
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    # DummyVecEnv  在当前进程中依次执行多个环境
    # SubprocVecEnv 把环境分配到不同子进程

    n = cfg.train.n_envs
    vec_cls = SubprocVecEnv if n > 1 else DummyVecEnv
    venv = make_vec_env(
        make_single_env(cfg, randomize=True),
        n_envs=n,
        seed=cfg.train.seed,
        vec_env_cls=vec_cls,
        # Monitor 负责统计 episode 信息
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
