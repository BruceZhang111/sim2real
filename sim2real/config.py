"""Typed configuration for the SO-101 sim2real pipeline.

Everything that changes behaviour lives here as plain dataclasses so a run is
fully described by one YAML file (see ``configs/``). The same config object is
consumed by training, evaluation, policy export and real-robot deployment,
which keeps sim and real perfectly in sync (observation layout, joint order,
action scaling, normalization, ...).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints, Union

import yaml

# Repository root (…/sim2real). Used to resolve asset/output paths so configs
# can stay relative and portable.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical joint order of the SO-101 follower arm (matches the MJCF actuators
# and the physical Feetech STS3215 servo IDs 1..6).
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT: str = "gripper"
ALL_JOINTS: tuple[str, ...] = ARM_JOINTS + (GRIPPER_JOINT,)


@dataclass
class EnvConfig:
    """Task + MuJoCo environment definition for the reach task."""

    # --- task / model / control ---
    env_id: str = "SO101Reach-v0"       # gym id -> selects the task env
    model_path: str = "assets/so101/reach_scene.xml"
    control_freq: float = 25.0          # Hz; realistic Feetech control-loop rate
    action_mode: str = "delta"          # "delta" | "absolute"
    action_scale: float = 0.1           # rad per control step (delta mode)
    # Optional faster delta scale for the gripper joint only (rad per control
    # step). Closing over a graspable object must take ~3 steps, not ~7 — the
    # STS3215 does ~5 rad/s so 0.25 rad per 40 ms tick is hardware-realistic.
    gripper_action_scale: float | None = None
    # Joints the policy actuates. Default: the 5 arm joints; the gripper is held
    # open for pure reaching. Add "gripper" here for grasping tasks.
    action_joints: list[str] = field(default_factory=lambda: list(ARM_JOINTS))
    gripper_hold: float = 0.5           # rad target for the (unactuated) gripper
    max_episode_steps: int = 200        # 200 * (1/25 Hz) = 8 s

    # --- reach task ---
    # Target sampling workspace box (metres, base frame). Tuned to the SO-101
    # reachable envelope (TCP reaches ~0.39 m forward at qpos=0).
    target_low: list[float] = field(default_factory=lambda: [0.12, -0.22, 0.06])
    target_high: list[float] = field(default_factory=lambda: [0.34, 0.22, 0.32])
    success_threshold: float = 0.02     # m; "reached" when TCP within this
    success_hold_steps: int = 10        # consecutive success steps to terminate
    init_qpos: list[float] | None = None  # home pose; None => zeros
    init_qpos_noise: float = 0.05       # rad uniform noise added to home

    # --- observation (all components are measurable on the real arm) ---
    obs_joint_noise_std: float = 0.0    # rad; proprio noise (raise for sim2real)
    obs_vel_noise_std: float = 0.0      # rad/s
    include_last_action: bool = True
    self_collision: bool = False        # disable arm self-collisions (keeps floor)
    # MuJoCo collides mesh geoms as convex hulls, and the vendored SO-101 jaw
    # meshes are concave — their hulls seal the gripper mouth shut, making any
    # grasp physically impossible. When True (default) the env rebuilds the two
    # jaw collision geoms as vertex-fitted boxes at load time (see base.py).
    fix_gripper_collision: bool = True
    max_joint_speed: float = 100.0      # rad/s; above this the sim is "diverged"
                                        # (blew up) -> end the episode with a penalty

    # --- reward weights ---
    w_dist: float = 1.0                 # -w_dist * ||tcp - target||
    w_near: float = 0.4                 # +w_near * exp(-dist / near_scale)
    near_scale: float = 0.05
    w_ctrl: float = 0.01                # -w_ctrl * ||action||^2
    w_vel: float = 0.0015               # -w_vel * ||qvel||^2  (smoothness)
    success_bonus: float = 5.0

    # --- pick-and-place task (ignored by the reach task) ---
    # NOTE: this task puts the cup pose in the observation, which the real arm
    # cannot measure without perception (camera / AprilTag). See docs.
    cup_init_low: list[float] = field(default_factory=lambda: [0.20, -0.12])
    cup_init_high: list[float] = field(default_factory=lambda: [0.28, 0.12])
    goal_low: list[float] = field(default_factory=lambda: [0.18, -0.14])
    goal_high: list[float] = field(default_factory=lambda: [0.30, 0.14])
    cup_rest_z: float = 0.025           # cup centre height resting on the table
    min_cup_goal_sep: float = 0.08      # ensure the goal isn't under the cup
    # Opposite-sides layout: the cup (on its red start pad) spawns on one side
    # of the workspace centreline and the green goal pad on the other, at
    # least side_min_y from the centreline; which side is which is random per
    # episode. Guarantees a real left-to-right (or right-to-left) transport.
    opposite_sides: bool = False
    side_min_y: float = 0.06
    grasp_dist: float = 0.035           # legacy TCP-cup "holding" radius (info only)
    lift_thresh: float = 0.02           # height above table counted as "lifted"
    lift_target: float = 0.12           # lift reward saturates here
    place_height_tol: float = 0.02      # cup within this of table height = placed
    cup_speed_thresh: float = 0.08      # cup must be near-stationary to count placed
    # "placed" only counts if the cup was lifted this high at some point during
    # the episode — otherwise shoving the cup onto the pad scores as success.
    min_lift_for_success: float = 0.06
    # Curriculum starts: fraction of TRAINING resets that begin mid-skill —
    # jaws around the cup (grasp_init_prob), already holding it
    # (hold_init_prob), or holding it just above the goal (place_init_prob,
    # teaches the descend-and-place endgame). All forced to 0 in eval envs
    # (env_factory) so success still measures the from-scratch task.
    grasp_init_prob: float = 0.0
    hold_init_prob: float = 0.0
    place_init_prob: float = 0.0
    # ... or with the open jaws a few cm BESIDE the cup (approach_init_prob):
    # the rung between "far away" and "jaws around the cup" that from-scratch
    # episodes otherwise have to discover unaided.
    approach_init_prob: float = 0.0
    # ... or holding the cup ELEVATED with the goal far away (carry_init_prob):
    # the transport rung. Place-starts begin at the goal and holding-starts on
    # the floor, so without this no episode ever demonstrates carrying.
    carry_init_prob: float = 0.0
    # Dense terms are POSITIVE shaped bonuses (1 - tanh(d/scale)); with an
    # all-negative dense reward, ending the episode early (flinging the cup out
    # of the workspace) outscores playing on, and PPO learns exactly that.
    w_reach: float = 1.0                # approach-the-cup bonus
    reach_scale: float = 0.06           # m; reach bonus length scale
    w_contact: float = 0.5              # per jaw-pad touching the cup (max 2 jaws)
    w_lift: float = 2.0                 # lift bonus (normalized by lift_target)
    # The lift bonus fades to zero within this XY distance of the goal —
    # otherwise lowering the cup onto the pad bleeds reward every step and the
    # policy learns to hover at altitude instead of placing.
    lift_fade_dist: float = 0.10
    w_grasp: float = 1.0                # holding bonus
    # Transport shaping uses the full 3D distance from the cup to the place
    # point (goal xy at rest height), so descending over the pad pays.
    w_transport: float = 1.5            # carry-to-place-point bonus while holding
    transport_scale: float = 0.10       # m; transport bonus length scale
    # "Holding" (grasp + transport income) requires the cup to be LIFTED —
    # otherwise dragging it along the floor pays like carrying — EXCEPT within
    # this 3D radius of the place point, where income continues through the
    # final centimetres of set-down (a lift-only gate is a reward cliff there).
    place_free_radius: float = 0.06
    # The cup holds fluid: tilting it spills. Tilt past spill_tilt (rad, angle
    # of the cup axis from vertical) ends the episode with a penalty — there
    # is no recovery from a spill. Dense uprightness shaping (w_upright *
    # (1 - cos(tilt))) gives gradient before the cliff, and "placed" requires
    # tilt < place_tilt. Yaw spin is allowed (it doesn't spill).
    spill_tilt: float = 0.45            # ~26 degrees
    place_tilt: float = 0.15            # ~9 degrees; upright enough to count
    w_upright: float = 1.0
    # Spill penalty (returned as-is; episode still terminates). −5 taught
    # policies to avoid the cup entirely — the termination already forfeits
    # future income, so keep the extra sting small.
    spill_penalty: float = -5.0

    def joint_names(self) -> list[str]:
        return list(ALL_JOINTS)


@dataclass
class DRConfig:
    """Domain randomization ranges — the heart of the sim2real transfer.

    Physics parameters are resampled every reset as ``nominal * U(low, high)``
    (from cached nominal values, so perturbations never compound). Latency and
    action noise model the real Feetech serial-bus command path.
    """

    enabled: bool = True
    # Multiplicative ranges applied per-reset to the *nominal* model values.
    mass_scale: tuple[float, float] = (0.8, 1.2)
    inertia_scale: tuple[float, float] = (0.8, 1.2)
    damping_scale: tuple[float, float] = (0.5, 1.6)
    frictionloss_scale: tuple[float, float] = (0.5, 1.6)
    armature_scale: tuple[float, float] = (0.7, 1.4)
    gain_scale: tuple[float, float] = (0.8, 1.2)    # actuator kp
    kv_scale: tuple[float, float] = (0.8, 1.2)      # actuator kv (damping)
    gravity_std: float = 0.15                       # m/s^2 added to g_z
    # Command-path effects (applied by wrappers).
    action_latency_steps: tuple[int, int] = (0, 2)  # random per-episode delay
    action_noise_std: float = 0.01                  # on the normalized action


@dataclass
class TrainConfig:
    """RL training hyper-parameters (Stable-Baselines3)."""

    algo: str = "ppo"                   # "ppo" | "sac"
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    seed: int = 0
    device: str = "auto"

    # PPO
    n_steps: int = 512
    batch_size: int = 1024
    n_epochs: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    learning_rate: float = 3e-4
    # Initial log-std of the Gaussian policy (PPO). The SB3 default (0.0 ->
    # std 1.0) is bang-bang exploration in a [-1,1] action space; a gripper
    # under that much noise cannot keep hold of anything, so grasp states never
    # acquire value. -1.0 (std 0.37) keeps early behaviour precise enough.
    log_std_init: float = 0.0

    # SAC
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    train_freq: int = 1
    tau: float = 0.005

    # shared
    policy_hidden: list[int] = field(default_factory=lambda: [256, 256])
    normalize_obs: bool = True
    normalize_reward: bool = True

    # bookkeeping
    eval_freq: int = 25_000             # per-env steps between evals
    n_eval_episodes: int = 20
    save_freq: int = 100_000
    log_dir: str = "outputs"
    run_name: str | None = None
    # Warm start: path to a checkpoint .zip whose policy weights (and, if a
    # sibling ckpt_vecnormalize_*.pkl exists, obs/reward normalization stats)
    # seed the new run. Useful when iterating on the reward of a learned skill.
    init_from: str | None = None


@dataclass
class DeployConfig:
    """Real-robot (Feetech STS3215) deployment settings."""

    port: str = "/dev/ttyACM0"
    baudrate: int = 1_000_000
    servo_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    # Feetech STS3215: 4096 ticks / revolution.
    ticks_per_rev: int = 4096
    # Per-joint centre tick (the tick that corresponds to 0 rad) and sign.
    # Defaults assume a symmetric mid-point calibration; override from your
    # LeRobot calibration file.
    center_ticks: list[int] = field(default_factory=lambda: [2048] * 6)
    joint_sign: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    max_step_rad: float = 0.15          # safety: max commanded delta per step
    torque_limit: float = 0.5           # fraction of max servo torque
    control_freq: float = 25.0


@dataclass
class Config:
    """Top-level config: one object fully describes a run."""

    env: EnvConfig = field(default_factory=EnvConfig)
    dr: DRConfig = field(default_factory=DRConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    # -- (de)serialization --------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        # tuples -> lists so PyYAML's SafeDumper can represent them; from_dict
        # restores tuple-typed fields on load, so the round-trip is lossless.
        return _to_plain(dataclasses.asdict(self))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


def _from_dict(dc_type: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys raise, so typos in a YAML config fail loudly instead of being
    silently ignored — important when a wrong DR range could quietly ruin a run.

    Type hints are resolved with ``get_type_hints`` so this works even under
    ``from __future__ import annotations`` (where ``field.type`` is a string).
    """
    if not is_dataclass(dc_type):
        return data
    hints = get_type_hints(dc_type)
    field_names = {f.name for f in fields(dc_type)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys for {dc_type.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name in field_names:
        if name not in data:
            continue
        val = data[name]
        hint = hints.get(name)
        if is_dataclass(hint) and isinstance(val, dict):
            kwargs[name] = _from_dict(hint, val)
        elif isinstance(val, list) and _is_tuple_hint(hint):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return dc_type(**kwargs)


def _to_plain(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML-safe serialization."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _is_tuple_hint(hint: Any) -> bool:
    """True if the (possibly Optional) type hint is a tuple type."""
    if get_origin(hint) is tuple or hint is tuple:
        return True
    if get_origin(hint) is Union:  # e.g. Optional[tuple[...]]
        return any(_is_tuple_hint(a) for a in get_args(hint))
    return False
