"""SO-101 reach task (MuJoCo + Gymnasium).

Move the tool-centre-point (TCP) to a randomly placed 3D target. This task is
**fully sim2real from proprioception alone** — every observation component is
available on the physical arm (joint angles/velocities from the servos, TCP via
forward kinematics, and the task-defined target). See :class:`SO101MujocoBase`
for the shared control/DR machinery.
"""

from __future__ import annotations

import numpy as np

from sim2real.config import ALL_JOINTS
from sim2real.envs.base import SO101MujocoBase


class SO101ReachEnv(SO101MujocoBase):
    """Reach a randomly placed 3D target with the SO-101 TCP."""

    def _setup_task(self) -> None:
        self._tcp_site_id = self._sid("gripperframe")
        self._target_body_id = self._bid("target")
        self._target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        if self._target_mocap_id < 0:
            raise RuntimeError("`target` body must be a mocap body in the scene XML")

    def _obs_dim(self) -> int:
        dim = len(ALL_JOINTS) * 2 + 3 + 3 + 3
        if self.cfg.include_last_action:
            dim += self.n_action
        return dim

    def _tcp(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site_id].copy()

    def _target(self) -> np.ndarray:
        return self.data.mocap_pos[self._target_mocap_id].copy()

    def _distance(self) -> float:
        return float(np.linalg.norm(self._tcp() - self._target()))

    def _reset_task(self) -> None:
        target = self.np_random.uniform(self.cfg.target_low, self.cfg.target_high)
        self.data.mocap_pos[self._target_mocap_id] = target

    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self._proprio()
        tcp, target = self._tcp(), self._target()
        parts = [qpos, qvel, tcp, target, target - tcp]
        if self.cfg.include_last_action:
            parts.append(self._last_action)
        return np.concatenate(parts).astype(np.float32)

    def _reach_reward(self, dist: float, action: np.ndarray) -> tuple[float, bool]:
        """Pure reward formula for a given TCP-target distance (uses live qvel)."""
        c = self.cfg
        qvel = self.data.qvel[self._vadr]
        reward = (
            -c.w_dist * dist
            + c.w_near * np.exp(-dist / c.near_scale)
            - c.w_ctrl * float(np.sum(action**2))
            - c.w_vel * float(np.sum(qvel**2))
        )
        success = dist < c.success_threshold
        if success:
            reward += c.success_bonus
        return reward, success

    def _reward_and_done(self, action):
        dist = self._distance()
        reward, success = self._reach_reward(dist, action)
        self._success_count = self._success_count + 1 if success else 0
        terminated = self._success_count >= self.cfg.success_hold_steps
        return reward, terminated, self._get_info(success)

    def _get_info(self, success: bool | None = None) -> dict:
        dist = self._distance()
        if success is None:
            success = dist < self.cfg.success_threshold
        return {"dist": dist, "is_success": bool(success),
                "tcp": self._tcp(), "target": self._target()}
