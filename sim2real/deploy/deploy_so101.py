"""Run a trained SO-101 reach policy on the real arm (or a MuJoCo stand-in).

    # sim closed loop (no hardware) — validates the whole deploy path:
    python -m sim2real.deploy.deploy_so101 --run outputs/ppo_reach_XXXX --target 0.25 0.05 0.2

    # real Feetech arm:
    python -m sim2real.deploy.deploy_so101 --run outputs/... --port /dev/ttyACM0 --target 0.25 0.0 0.2

The observation is rebuilt to byte-for-byte match training (same joint order,
FK-derived TCP, last-action tail), the baked-normalization ONNX policy produces
an action, and safety clamps (max step, joint limits) guard the servos before
anything is written to the bus.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sim2real.config import ALL_JOINTS, Config, REPO_ROOT
from sim2real.deploy.feetech_bus import Calibration, FeetechBus, SimPlantBus
from sim2real.utils import ForwardKinematics


class PolicyController:
    """Closed-loop controller shared by sim and hardware backends."""

    def __init__(self, run_dir: Path, bus, cfg: Config):
        import json
        import onnxruntime as ort

        self.cfg = cfg
        self.bus = bus
        self.cal = Calibration(cfg.deploy)
        self.fk = ForwardKinematics()
        self.ranges = self.fk.joint_ranges  # (6, 2)

        meta = json.loads((run_dir / "policy_meta.json").read_text())
        self.meta = meta
        self.obs_dim = meta["obs_dim"]
        self.action_joints = meta["action_joints"]
        self.action_idx = np.array([ALL_JOINTS.index(j) for j in self.action_joints], dtype=int)
        self.action_mode = meta["action_mode"]
        # scalar in old metas, per-joint list in new ones — both broadcast in
        # _action_to_targets
        self.action_scale = np.asarray(meta["action_scale"], dtype=np.float64)
        self.include_last_action = meta["include_last_action"]
        self.dt = 1.0 / cfg.deploy.control_freq

        self.session = ort.InferenceSession(
            str(run_dir / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        self._last_action = np.zeros(len(self.action_joints), dtype=np.float32)
        self._last_qpos = None

    def _build_obs(self, qpos: np.ndarray, qvel: np.ndarray, target: np.ndarray) -> np.ndarray:
        tcp = self.fk.tcp(qpos)
        parts = [qpos, qvel, tcp, target, target - tcp]
        if self.include_last_action:
            parts.append(self._last_action)
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == self.obs_dim, f"obs dim {obs.shape[0]} != {self.obs_dim}"
        return obs[None, :]

    def _action_to_targets(self, action: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Map policy action -> full 6-joint position targets (radians), safely."""
        target_q = qpos.copy()
        lo = self.ranges[self.action_idx, 0]
        hi = self.ranges[self.action_idx, 1]
        cur = qpos[self.action_idx]
        if self.action_mode == "delta":
            desired = cur + action * self.action_scale
        else:  # absolute
            desired = lo + (action + 1.0) * 0.5 * (hi - lo)
        # safety: bound per-step motion, then joint limits
        step = np.clip(desired - cur, -self.cfg.deploy.max_step_rad, self.cfg.deploy.max_step_rad)
        target_q[self.action_idx] = np.clip(cur + step, lo, hi)
        # hold the gripper (and any non-actuated joint) at the configured pose
        g = ALL_JOINTS.index("gripper")
        if g not in self.action_idx:
            target_q[g] = np.clip(self.cfg.env.gripper_hold, self.ranges[g, 0], self.ranges[g, 1])
        return target_q

    def step(self, target: np.ndarray) -> dict:
        ticks = self.bus.read_ticks()
        qpos = self.cal.ticks_to_rad(ticks)
        if self._last_qpos is None:
            self._last_qpos = qpos
        qvel = (qpos - self._last_qpos) / self.dt
        self._last_qpos = qpos

        obs = self._build_obs(qpos, qvel, target)
        action = self.session.run(["action"], {"obs": obs})[0][0]
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.astype(np.float32)

        target_q = self._action_to_targets(action, qpos)
        self.bus.write_ticks(self.cal.rad_to_ticks(target_q))

        tcp = self.fk.tcp(qpos)
        return {"dist": float(np.linalg.norm(tcp - target)), "tcp": tcp, "qpos": qpos}

    def run(self, target: np.ndarray, max_steps: int, success_threshold: float) -> None:
        print(f"[deploy] target={np.round(target,3)}  control@{self.cfg.deploy.control_freq}Hz")
        if hasattr(self.bus, "set_target"):
            self.bus.set_target(target)
        for t in range(max_steps):
            t0 = time.time()
            info = self.step(target)
            if t % 10 == 0 or info["dist"] < success_threshold:
                print(f"  step {t:3d}  dist={info['dist']*1000:6.1f} mm  tcp={np.round(info['tcp'],3)}")
            if info["dist"] < success_threshold:
                print(f"[deploy] reached target in {t} steps ({info['dist']*1000:.1f} mm)")
                return
            # real-time pacing (sim backend ignores this budget cheaply)
            sleep = self.dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
        print(f"[deploy] finished {max_steps} steps, final dist {info['dist']*1000:.1f} mm")


def build_bus(cfg: Config, port: str | None):
    if port:
        dep = cfg.deploy
        dep.port = port
        bus = FeetechBus(dep)
        bus.connect()
        bus.enable_torque(True)
        print(f"[deploy] connected to real arm on {port}")
        return bus
    bus = SimPlantBus(cfg.deploy)
    bus.connect()
    print("[deploy] no --port given: using MuJoCo SimPlantBus (no hardware)")
    return bus


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy SO-101 reach policy")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--port", type=str, default=None, help="serial port for the real arm")
    p.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--steps", type=int, default=150)
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    cfg = Config.from_yaml(run_dir / "config.yaml")

    if args.target is None:
        lo, hi = np.array(cfg.env.target_low), np.array(cfg.env.target_high)
        target = (lo + hi) / 2.0
    else:
        target = np.array(args.target, dtype=np.float64)

    bus = build_bus(cfg, args.port)
    ctrl = PolicyController(run_dir, bus, cfg)
    try:
        ctrl.run(target, args.steps, cfg.env.success_threshold)
    except KeyboardInterrupt:
        print("\n[deploy] interrupted")
    finally:
        bus.disconnect()
        print("[deploy] bus closed, torque disabled")


if __name__ == "__main__":
    main()
