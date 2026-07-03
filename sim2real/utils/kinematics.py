"""Forward kinematics for the SO-101, backed by the same MuJoCo model used in
simulation.

Deployment on the real arm needs the tool-centre-point (TCP) position to build
the observation, but the Feetech servos only report joint angles. Rather than
hand-derive a DH chain (and risk sim/real mismatch), we evaluate FK with the
*identical* MJCF used for training. One source of truth => no transfer gap from
kinematics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim2real.config import ALL_JOINTS, REPO_ROOT

# The bare robot (no floor/target) is enough for FK and avoids needing the
# scene wrapper.
ROBOT_XML = REPO_ROOT / "assets" / "so101" / "so101_new_calib.xml"
TCP_SITE = "gripperframe"


class ForwardKinematics:
    """Evaluate SO-101 TCP position/orientation from joint angles."""

    def __init__(self, xml_path: str | Path | None = None, tcp_site: str = TCP_SITE):
        import mujoco  # local import: keeps `sim2real.config` import cheap

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or ROBOT_XML))
        self.data = mujoco.MjData(self.model)
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
        if self.tcp_site_id < 0:
            raise ValueError(f"TCP site {tcp_site!r} not found in {xml_path or ROBOT_XML}")
        # Map joint name -> qpos address (all SO-101 joints are single-dof hinges).
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ALL_JOINTS]
        self._qadr = np.array([self.model.jnt_qposadr[j] for j in jids], dtype=int)
        # (6, 2) joint limits in ALL_JOINTS order — used by deploy for safety.
        self.joint_ranges = self.model.jnt_range[jids].copy()

    def tcp(self, qpos: np.ndarray) -> np.ndarray:
        """Return the TCP position (x, y, z) in the base frame for 6 joint angles."""
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos.shape[0] != len(ALL_JOINTS):
            raise ValueError(f"expected {len(ALL_JOINTS)} joint angles, got {qpos.shape[0]}")
        self.data.qpos[self._qadr] = qpos
        self._mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.tcp_site_id].copy()

    def tcp_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (position (3,), rotation matrix (3, 3)) of the TCP."""
        pos = self.tcp(qpos)
        rot = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        return pos, rot
