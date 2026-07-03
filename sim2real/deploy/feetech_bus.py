"""Servo bus abstraction for SO-101 deployment.

Two implementations share one interface so the exact same control loop runs on
hardware or in simulation:

* :class:`FeetechBus`  – real STS3215 servos over the Feetech serial SDK.
* :class:`SimPlantBus` – a MuJoCo model standing in for the robot, so the whole
  deploy pipeline (obs build, FK, tick<->rad conversion, safety clamps) can be
  exercised and tested without the arm plugged in.

Angle convention: 0 rad at ``center_ticks``; ``ticks_per_rev`` ticks per full
turn; per-joint ``joint_sign`` flips direction to match the URDF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim2real.config import ALL_JOINTS, DeployConfig, REPO_ROOT

TWO_PI = 2.0 * np.pi


class Calibration:
    """Convert between servo ticks and joint radians."""

    def __init__(self, cfg: DeployConfig):
        self.center = np.asarray(cfg.center_ticks, dtype=np.float64)
        self.sign = np.asarray(cfg.joint_sign, dtype=np.float64)
        self.tpr = float(cfg.ticks_per_rev)

    def ticks_to_rad(self, ticks: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(ticks, dtype=np.float64) - self.center) * TWO_PI / self.tpr

    def rad_to_ticks(self, rad: np.ndarray) -> np.ndarray:
        ticks = self.center + self.sign * np.asarray(rad, dtype=np.float64) * self.tpr / TWO_PI
        return np.round(ticks).astype(np.int64)


class BusInterface:
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def enable_torque(self, on: bool) -> None: ...
    def read_ticks(self) -> np.ndarray: raise NotImplementedError
    def write_ticks(self, ticks: np.ndarray) -> None: raise NotImplementedError


class FeetechBus(BusInterface):
    """Real STS3215 bus via the ``scservo_sdk`` (feetech-servo-sdk) package."""

    ADDR_TORQUE_ENABLE = 40
    ADDR_GOAL_POSITION = 42
    ADDR_PRESENT_POSITION = 56
    LEN_POSITION = 2

    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.ids = list(cfg.servo_ids)
        self._scs = None
        self._port = None
        self._packet = None

    def connect(self) -> None:
        import scservo_sdk as scs  # lazy: only needed on the real robot

        self._scs = scs
        self._port = scs.PortHandler(self.cfg.port)
        self._packet = scs.PacketHandler(0)  # STS/SCS protocol
        if not self._port.openPort():
            raise IOError(f"failed to open {self.cfg.port}")
        if not self._port.setBaudRate(self.cfg.baudrate):
            raise IOError(f"failed to set baudrate {self.cfg.baudrate}")

    def disconnect(self) -> None:
        if self._port is not None:
            self.enable_torque(False)
            self._port.closePort()

    def enable_torque(self, on: bool) -> None:
        for i in self.ids:
            self._packet.write1ByteTxRx(self._port, i, self.ADDR_TORQUE_ENABLE, 1 if on else 0)

    def read_ticks(self) -> np.ndarray:
        scs = self._scs
        reader = scs.GroupSyncRead(
            self._port, self._packet, self.ADDR_PRESENT_POSITION, self.LEN_POSITION
        )
        for i in self.ids:
            reader.addParam(i)
        reader.txRxPacket()
        out = []
        for i in self.ids:
            out.append(reader.getData(i, self.ADDR_PRESENT_POSITION, self.LEN_POSITION))
        return np.asarray(out, dtype=np.int64)

    def write_ticks(self, ticks: np.ndarray) -> None:
        scs = self._scs
        writer = scs.GroupSyncWrite(
            self._port, self._packet, self.ADDR_GOAL_POSITION, self.LEN_POSITION
        )
        for i, t in zip(self.ids, np.asarray(ticks, dtype=np.int64)):
            t = int(t) & 0xFFFF
            param = [scs.SCS_LOBYTE(t), scs.SCS_HIBYTE(t)]
            writer.addParam(i, param)
        writer.txPacket()


class SimPlantBus(BusInterface):
    """MuJoCo-backed stand-in robot (no hardware needed).

    Behaves like a servo bus but the "present position" comes from a MuJoCo
    simulation stepped by the position targets. Perfect for validating the
    deploy code path and for a fully sim closed loop.
    """

    def __init__(self, cfg: DeployConfig, model_path: str | Path | None = None):
        import mujoco

        self._mj = mujoco
        self.cfg = cfg
        self.cal = Calibration(cfg)
        path = Path(model_path or (REPO_ROOT / "assets" / "so101" / "reach_scene.xml"))
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._qadr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in ALL_JOINTS], dtype=int,
        )
        self._act = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ALL_JOINTS],
            dtype=int,
        )
        sim_dt = self.model.opt.timestep
        self.frame_skip = max(1, round((1.0 / cfg.control_freq) / sim_dt))

    def connect(self) -> None:
        self._mj.mj_resetData(self.model, self.data)
        self._mj.mj_forward(self.model, self.data)

    def read_ticks(self) -> np.ndarray:
        rad = self.data.qpos[self._qadr]
        return self.cal.rad_to_ticks(rad)

    def write_ticks(self, ticks: np.ndarray) -> None:
        rad = self.cal.ticks_to_rad(ticks)
        self.data.ctrl[self._act] = rad
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

    # convenience for the sim closed loop: place the visual target
    def set_target(self, xyz) -> None:
        bid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "target")
        mid = int(self.model.body_mocapid[bid])
        if mid >= 0:
            self.data.mocap_pos[mid] = np.asarray(xyz, dtype=np.float64)
