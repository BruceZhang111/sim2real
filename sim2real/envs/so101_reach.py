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

    # 在 MuJoCo 模型中查找已有对象，并保存它们的 ID
    def _setup_task(self) -> None:
        self._tcp_site_id = self._sid("gripperframe")
        # target在body中有ID，但并非所有的body都是mocap(motion capture body)，所以还要找到target对应的mocap ID
        """普通 body：
        受到重力、碰撞和动力学影响
        位置由 MuJoCo 计算

        mocap body：
        位置由程序直接指定
        不参与普通动力学积分
        
        body_id 用来访问 body 相关数据；
        mocap_id 用来访问 mocap 位置和姿态数据；
        """
        self._target_body_id = self._bid("target")
        self._target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        if self._target_mocap_id < 0:
            raise RuntimeError("`target` body must be a mocap body in the scene XML")

    def _obs_dim(self) -> int:
        """关节位置 qpos       len(ALL_JOINTS)
            关节速度 qvel       len(ALL_JOINTS)
            TCP 位置             3
            目标位置              3
            目标相对 TCP 的位移   3
            上一次 action         n_action，可选"""
        dim = len(ALL_JOINTS) * 2 + 3 + 3 + 3
        if self.cfg.include_last_action:
            dim += self.n_action
        return dim

    def _tcp(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site_id].copy()

    def _target(self) -> np.ndarray:
        # Reach 环境的目标位置不是通过 body 的普通 xpos 读取，而是通过 mocap 数据读取
        return self.data.mocap_pos[self._target_mocap_id].copy()

    def _distance(self) -> float:
        # 计算 TCP 与目标之间的距离
        return float(np.linalg.norm(self._tcp() - self._target()))

    def _reset_task(self) -> None:
        """
        target_low  = [x_min, y_min, z_min]
        target_high = [x_max, y_max, z_max]
        """
        target = self.np_random.uniform(self.cfg.target_low, self.cfg.target_high)
        self.data.mocap_pos[self._target_mocap_id] = target

    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self._proprio()
        tcp, target = self._tcp(), self._target()
        parts = [qpos, qvel, tcp, target, target - tcp]
        if self.cfg.include_last_action:
            parts.append(self._last_action)
        # np.concatenate(parts) 会沿着第 0 维，把它们首尾拼接成一个完整的一维数组
        return np.concatenate(parts).astype(np.float32)

    def _reach_reward(self, dist: float, action: np.ndarray) -> tuple[float, bool]:
        """Pure reward formula for a given TCP-target distance (uses live qvel)."""
        c = self.cfg
        qvel = self.data.qvel[self._vadr]
        """
        总奖励 =
        接近目标的奖励
        - 距离惩罚
        - 控制能量惩罚
        - 运动速度惩罚
        """
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
        # 连续成功计数，失败时归零
        self._success_count = self._success_count + 1 if success else 0
        terminated = self._success_count >= self.cfg.success_hold_steps
        return reward, terminated, self._get_info(success)

    def _get_info(self, success: bool | None = None) -> dict:
        dist = self._distance()
        if success is None:
            success = dist < self.cfg.success_threshold
        return {"dist": dist, "is_success": bool(success),
                "tcp": self._tcp(), "target": self._target()}
