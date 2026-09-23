"""Command-path domain randomization wrappers.

These model the imperfect path between the policy's action and the servo
actually moving on the real arm: a few control-steps of serial-bus / control
latency, and small command noise. Randomizing them in sim makes the learned
policy robust to the real robot's timing jitter.
"""
# 模拟动作延迟和动作噪声的动作输出类，对base等父类做进一步封装
from __future__ import annotations

import numpy as np
import gymnasium as gym


class ActionLatencyWrapper(gym.Wrapper):
    """Delay applied actions by k control steps; k is resampled each episode."""
    # 通过建立零动作占位的方法实现动作延迟
    def __init__(self, env, min_steps: int = 0, max_steps: int = 2):
        super().__init__(env)
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self._k = 0
        # list[np.ndarray] 是类型标注
        self._buf: list[np.ndarray] = []
        self._zero = np.zeros(env.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rng = self.env.unwrapped.np_random
        # 从 [min_steps, max_steps] 含两端均匀抽取一个整数 k
        self._k = int(rng.integers(self.min_steps, self.max_steps + 1))
        # buf中包含k个独立的零动作数组
        self._buf = [self._zero.copy() for _ in range(self._k)]
        return obs, info

    def step(self, action):
        if self._k == 0:
            return self.env.step(action)
        self._buf.append(np.asarray(action, dtype=np.float32))
        applied = self._buf.pop(0)
        return self.env.step(applied)


class ActionNoiseWrapper(gym.Wrapper):
    """Add Gaussian noise to the action (models command quantization/jitter)."""
    # 给动作引入高斯噪声
    def __init__(self, env, std: float = 0.0):
        super().__init__(env)
        self.std = float(std)

    def step(self, action):
        if self.std > 0:
            rng = self.env.unwrapped.np_random
            action = np.asarray(action, dtype=np.float32) + rng.normal(
                0.0, self.std, size=self.env.action_space.shape
            )
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return self.env.step(action)
