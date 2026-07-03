"""Pipeline tests — run on CPU in a few seconds, no GPU or hardware needed.

These lock down the sim2real-critical invariants: deterministic resets, an
observation that stays reproducible, domain randomization that varies but never
compounds, and FK/normalization/calibration parity between sim and deploy.
"""

import numpy as np
import gymnasium as gym
import pytest

import sim2real  # noqa: F401  (registers SO101Reach-v0)
from sim2real.config import ALL_JOINTS, Config, DeployConfig, DRConfig, EnvConfig
from sim2real.env_factory import make_single_env


def make(dr=False, **env_over):
    cfg = EnvConfig(**env_over)
    return gym.make("SO101Reach-v0", config=cfg, dr=DRConfig(enabled=dr))


# ---------------------------------------------------------------- config
def test_config_roundtrip(tmp_path):
    cfg = Config()
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert Config.from_yaml(path).to_dict() == cfg.to_dict()


def test_config_unknown_key_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"env": {"not_a_real_field": 1}})


def test_config_tuple_fields_restored():
    cfg = Config.from_dict({"dr": {"mass_scale": [0.5, 1.5]}})
    assert isinstance(cfg.dr.mass_scale, tuple)


# ---------------------------------------------------------------- env basics
def test_spaces_and_reset():
    env = make()
    assert env.action_space.shape == (5,)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (26,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert "dist" in info and "is_success" in info
    env.close()


def test_determinism():
    env = make()
    env.reset(seed=7)
    o1 = env.step(np.zeros(5, np.float32))[0]
    env.reset(seed=7)
    o2 = env.step(np.zeros(5, np.float32))[0]
    assert np.allclose(o1, o2)
    env.close()


def test_step_return_types():
    env = make()
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    env.close()


def test_truncation_at_horizon():
    env = make(max_episode_steps=5)
    env.reset(seed=0)
    trunc = False
    for _ in range(5):
        trunc = env.step(np.zeros(5, np.float32))[3]
    assert trunc
    env.close()


def test_reward_prefers_closer():
    u = make().unwrapped
    u.reset(seed=0)
    r_near, _ = u._reach_reward(0.0, np.zeros(5))
    r_far, _ = u._reach_reward(0.3, np.zeros(5))
    assert r_near > r_far


def test_success_when_target_at_tcp():
    env = make(success_threshold=0.05)
    env.reset(seed=1)
    u = env.unwrapped
    tcp = u.data.site_xpos[u._tcp_site_id].copy()
    u.data.mocap_pos[u._target_mocap_id] = tcp
    _, _, _, _, info = env.step(np.zeros(5, np.float32))
    assert info["dist"] < 0.05
    assert info["is_success"]
    env.close()


# ---------------------------------------------------------------- domain rand
def test_dr_varies_but_does_not_compound():
    u = make(dr=True).unwrapped
    u.reset(seed=1)
    a = u.model.body_mass.copy()
    u.reset(seed=2)
    b = u.model.body_mass.copy()
    assert not np.allclose(a, b), "DR should change masses across resets"

    nominal = u._nominal["body_mass"]
    mask = nominal > 0
    for _ in range(6):  # repeated resets must stay within the configured band
        u.reset()
        ratio = u.model.body_mass[mask] / nominal[mask]
        assert ratio.min() >= u.dr.mass_scale[0] - 1e-6
        assert ratio.max() <= u.dr.mass_scale[1] + 1e-6


def test_dr_disabled_restores_nominal():
    u = make(dr=False).unwrapped
    u.reset(seed=1)
    assert np.allclose(u.model.body_mass, u._nominal["body_mass"])
    assert np.allclose(u.model.dof_damping, u._nominal["dof_damping"])


def test_self_collision_flag():
    # With self_collision=False no two arm geoms may collide; with True they can.
    def arm_pair_collides(u):
        arm = []
        for g in range(u.model.ngeom):
            ct, ca = u.model.geom_contype[g], u.model.geom_conaffinity[g]
            if ct == 0 and ca == 0:
                continue  # visual-only
            b = u.model.geom_bodyid[g]
            name = u._mj.mj_id2name(u.model, u._mj.mjtObj.mjOBJ_BODY, b)
            if name in ("world", "target") or u._body_has_free_joint(b):
                continue
            arm.append((ct, ca))
        for i in range(len(arm)):
            for j in range(i + 1, len(arm)):
                if (arm[i][0] & arm[j][1]) or (arm[j][0] & arm[i][1]):
                    return True
        return False
    assert not arm_pair_collides(make(self_collision=False).unwrapped)
    assert arm_pair_collides(make(self_collision=True).unwrapped)


# ---------------------------------------------------------------- FK parity
def test_forward_kinematics_matches_sim():
    from sim2real.utils import ForwardKinematics
    env = make()
    env.reset(seed=3)
    u = env.unwrapped
    q = u.data.qpos[u._qadr].copy()
    fk = ForwardKinematics()
    assert np.allclose(fk.tcp(q), u.data.site_xpos[u._tcp_site_id], atol=1e-3)
    assert fk.joint_ranges.shape == (6, 2)
    env.close()


# ---------------------------------------------------------------- wrappers
class _RecordingEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (3,), np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.applied = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.applied.append(np.asarray(action, np.float32).copy())
        return np.zeros(1, np.float32), 0.0, False, False, {}


def test_action_latency_delays_by_k():
    from sim2real.wrappers import ActionLatencyWrapper
    base = _RecordingEnv()
    env = ActionLatencyWrapper(base, 2, 2)
    env.reset(seed=0)
    a1 = np.ones(3, np.float32)
    env.step(a1)
    env.step(np.full(3, 0.5, np.float32))
    assert np.allclose(base.applied[0], 0) and np.allclose(base.applied[1], 0)
    env.step(np.full(3, -1, np.float32))
    assert np.allclose(base.applied[2], a1)  # a1 emerges after a 2-step delay


def test_action_noise_stays_in_bounds():
    from sim2real.wrappers import ActionNoiseWrapper
    base = _RecordingEnv()
    env = ActionNoiseWrapper(base, std=0.5)
    env.reset(seed=0)
    for _ in range(30):
        env.step(np.ones(3, np.float32))
    applied = np.array(base.applied)
    assert applied.max() <= 1.0 + 1e-6 and applied.min() >= -1.0 - 1e-6

    base0 = _RecordingEnv()
    env0 = ActionNoiseWrapper(base0, std=0.0)
    env0.reset(seed=0)
    env0.step(np.full(3, 0.3, np.float32))
    assert np.allclose(base0.applied[-1], 0.3)  # std=0 is identity


# ---------------------------------------------------------------- factory / deploy
def test_factory_builds_and_steps():
    cfg = Config()
    env = make_single_env(cfg, randomize=True)()
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()


# ---------------------------------------------------------------- pick & place
def make_pp(**over):
    from sim2real.config import ARM_JOINTS, GRIPPER_JOINT
    cfg = EnvConfig(
        env_id="SO101PickPlace-v0",
        model_path="assets/so101/pickplace_scene.xml",
        action_joints=list(ARM_JOINTS) + [GRIPPER_JOINT],
        **over,
    )
    return gym.make("SO101PickPlace-v0", config=cfg, dr=DRConfig(enabled=False))


def test_pickplace_spaces_and_obs():
    env = make_pp()
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (6,)       # gripper is actuated
    assert obs.shape == (33,)
    assert {"dist", "tcp_to_cup", "lift", "holding", "is_success"} <= set(info)
    env.close()


def test_pickplace_gripper_is_actuated():
    u = make_pp().unwrapped
    assert u.n_action == 6 and len(u.hold_joint_idx) == 0


def test_pickplace_cup_goal_separation():
    u = make_pp().unwrapped
    for s in range(5):
        u.reset(seed=s)
        assert np.linalg.norm(u._cup()[:2] - u._goal()[:2]) >= u.cfg.min_cup_goal_sep - 1e-6


def test_pickplace_determinism():
    env = make_pp()
    env.reset(seed=3)
    o1 = env.step(np.zeros(6, np.float32))[0]
    env.reset(seed=3)
    o2 = env.step(np.zeros(6, np.float32))[0]
    assert np.allclose(o1, o2)
    env.close()


def test_jaw_collision_pads_replace_hulled_meshes():
    # MuJoCo collides meshes as convex hulls; the concave SO-101 jaw meshes
    # would seal the gripper mouth shut. base._load_model must disable them and
    # add box pads instead (fix_gripper_collision=True default).
    import mujoco
    u = make_pp().unwrapped
    m = u.model
    pad_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                 for g in range(m.ngeom)]
    for body in ("gripper", "moving_jaw_so101_v1"):
        pads = [n for n in pad_names if n.startswith(f"{body}_pad")]
        assert len(pads) >= 3, f"missing jaw pads on {body}"
        # the hulled jaw meshes must no longer collide
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and m.geom_dataid[g] >= 0:
                mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
                if mesh in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
                    assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


def test_gripper_can_hold_cup():
    # End-to-end grasp physics: fingers-down pose over the cup, close the jaws,
    # lift the arm -> the cup must come along. This is the invariant the whole
    # pick-place task rests on (it fails with the raw hulled jaw meshes).
    import mujoco
    u = make_pp().unwrapped
    u.reset(seed=0)
    m, d = u.model, u.data
    q_grasp = np.array([-0.0009, 0.2208, 0.1735, 1.1757, 0.0209])  # TCP over cup, fingers down
    arm_idx, g_qadr = u._qadr[:5], u._qadr[5]
    arm_act, g_act = u._act_ids[:5], u._act_ids[5]
    d.qpos[arm_idx] = q_grasp
    d.qpos[g_qadr] = 1.2
    d.qvel[:] = 0.0
    d.ctrl[arm_act], d.ctrl[g_act] = q_grasp, 1.2
    d.qpos[u._cup_qadr] = [0.27, 0.0, u.cfg.cup_rest_z, 1, 0, 0, 0]
    d.qvel[u._cup_vadr] = 0.0
    mujoco.mj_forward(m, d)
    for gtar in np.linspace(1.2, -0.17, 40):        # close
        d.ctrl[g_act] = gtar
        for _ in range(10):
            mujoco.mj_step(m, d)
    q_up = q_grasp.copy()
    q_up[1] -= 0.45                                  # lift via shoulder
    for a in np.linspace(0, 1, 60):
        d.ctrl[arm_act] = (1 - a) * q_grasp + a * q_up
        for _ in range(8):
            mujoco.mj_step(m, d)
    lift = d.xpos[u._cup_bid][2] - u.cfg.cup_rest_z
    assert lift > 0.05, f"cup not held (lift={lift*1000:.1f} mm)"


def test_opposite_sides_layout():
    # cup (red pad) and goal (green pad) must spawn on opposite sides of the
    # centreline, both at least side_min_y out, and the red pad tracks the cup
    u = make_pp(opposite_sides=True, side_min_y=0.06).unwrapped
    for s in range(6):
        u.reset(seed=s)
        cy, gy = u._cup()[1], u._goal()[1]
        assert cy * gy < 0, "cup and goal on the same side"
        assert abs(cy) >= 0.05 and abs(gy) >= 0.05
        pad = u.data.mocap_pos[u._start_mocap_id]
        assert np.linalg.norm(pad[:2] - u._cup()[:2]) < 1e-6


def test_pads_never_coincide_in_any_start_mode():
    # the red and green circles must be distinct places in every episode type
    import dataclasses
    probs = [f.name for f in dataclasses.fields(EnvConfig())
             if f.name.endswith("_init_prob")]
    for mode in probs:
        env = make_pp(opposite_sides=True, **{mode: 1.0})
        u = env.unwrapped
        for s in range(4):
            u.reset(seed=s)
            pad = u.data.mocap_pos[u._start_mocap_id][:2]
            goal = u._goal()[:2]
            assert np.linalg.norm(pad - goal) > 0.03, \
                f"{mode}: red and green pads coincide"
        env.close()


def test_eval_env_zeroes_all_curriculum_probs():
    # Eval must measure the from-scratch task: every *_init_prob field has to
    # be zeroed by the factory (a forgotten new rung silently inflates eval
    # success by exactly its probability — this happened).
    import dataclasses
    from sim2real.config import ARM_JOINTS, GRIPPER_JOINT
    cfg = Config()
    cfg.env.env_id = "SO101PickPlace-v0"
    cfg.env.model_path = "assets/so101/pickplace_scene.xml"
    cfg.env.action_joints = list(ARM_JOINTS) + [GRIPPER_JOINT]
    prob_fields = [f.name for f in dataclasses.fields(cfg.env)
                   if f.name.endswith("_init_prob")]
    assert prob_fields, "expected curriculum prob fields on EnvConfig"
    for name in prob_fields:
        setattr(cfg.env, name, 0.5)
    env = make_single_env(cfg, randomize=False)()
    for name in prob_fields:
        assert getattr(env.unwrapped.cfg, name) == 0.0, f"{name} leaked into eval"
    env.close()


def test_tilted_cup_spills_and_cannot_be_placed():
    # The cup holds fluid: tilting past spill_tilt ends the episode, and a cup
    # lying on its side at the goal must never count as "placed" (its centre
    # height alone would pass the place_height_tol check).
    import mujoco
    u = make_pp().unwrapped
    u.reset(seed=0)
    goal = u._goal()
    # cup on its side exactly at the goal, pretending it was lifted earlier
    u.data.qpos[u._cup_qadr] = [goal[0], goal[1], 0.014,
                                np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    u.data.qvel[u._cup_vadr] = 0.0
    u._max_lift = 1.0
    mujoco.mj_forward(u.model, u.data)
    *_, placed = u._metrics()
    assert not placed, "a tipped-over cup counted as placed"
    reward, terminated, info = u._reward_and_done(np.zeros(6))
    assert terminated and info.get("spilled"), "tilt beyond spill_tilt must end the episode"


def test_calibration_roundtrip():
    from sim2real.deploy import Calibration
    cal = Calibration(DeployConfig())
    rad = np.array([0.1, -0.2, 0.3, 0.0, 0.5, 0.4])
    back = cal.ticks_to_rad(cal.rad_to_ticks(rad))
    assert np.allclose(back, rad, atol=2 * np.pi / 4096)  # within one tick


def test_sim_plant_bus_read_write():
    from sim2real.deploy import SimPlantBus
    bus = SimPlantBus(DeployConfig())
    bus.connect()
    ticks = bus.read_ticks()
    assert ticks.shape == (6,)
    bus.write_ticks(ticks)  # steps the sim without raising
    assert bus.read_ticks().shape == (6,)
