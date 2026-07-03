"""Shared MuJoCo env machinery for SO-101 tasks.

Both the reach and pick-place environments subclass :class:`SO101MujocoBase`,
which owns everything task-agnostic: model loading, control-rate handling,
position-target action mapping (delta/absolute), dynamics domain randomization
(cached, non-compounding), collision configuration, and rendering. Subclasses
implement the task hooks: ``_setup_task``, ``_reset_task``, ``_get_obs``,
``_get_info`` and ``_reward_and_done``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim2real.config import ALL_JOINTS, DRConfig, EnvConfig, REPO_ROOT

# Gripper collision fix (cfg.fix_gripper_collision).
#
# MuJoCo collides mesh geoms as their CONVEX HULLS. The vendored SO-101 model
# reuses the concave visual meshes as collision geoms, so the hull of the
# palm+fixed-jaw piece and the hull of the moving jaw fill the space between
# the jaws: in collision space the gripper mouth is solid and nothing can ever
# be grasped (objects get expelled from the hull volume with deep soft
# penetration). We keep the vendored MJCF pristine and instead rebuild those
# two geoms at load time via MjSpec: collision is disabled on the hulled mesh
# and replaced by boxes fitted to the mesh vertices — the finger blade is split
# into slabs along its long axis (to follow the taper) plus one box for the
# palm/hinge base.
#
# body name -> (collision mesh, blade long axis in body frame, blade extent:
#               vertices with coord[axis] < limit belong to the finger blade)
_JAW_COLLISION_FIX = {
    "gripper": ("wrist_roll_follower_so101_v1", 2, -0.030),
    "moving_jaw_so101_v1": ("moving_jaw_so101_v1", 1, -0.010),
}


class SO101MujocoBase(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config=None, dr=None, render_mode=None):
        import mujoco

        self._mj = mujoco
        self.cfg = config or EnvConfig()
        self.dr = dr or DRConfig()
        self.render_mode = render_mode

        model_path = Path(self.cfg.model_path)
        if not model_path.is_absolute():
            model_path = REPO_ROOT / model_path
        self.model = self._load_model(str(model_path))
        self.data = mujoco.MjData(self.model)

        sim_dt = self.model.opt.timestep
        self.frame_skip = max(1, round((1.0 / self.cfg.control_freq) / sim_dt))
        self.metadata = dict(self.metadata, render_fps=int(round(self.cfg.control_freq)))

        # joint / actuator bookkeeping (by name -> robust to extra object dofs)
        self._joint_ids = np.array([self._jid(j) for j in ALL_JOINTS], dtype=int)
        self._qadr = self.model.jnt_qposadr[self._joint_ids].copy()
        self._vadr = self.model.jnt_dofadr[self._joint_ids].copy()
        self._jnt_range = self.model.jnt_range[self._joint_ids].copy()
        self._act_ids = np.array([self._aid(j) for j in ALL_JOINTS], dtype=int)

        self.action_joint_idx = np.array(
            [ALL_JOINTS.index(j) for j in self.cfg.action_joints], dtype=int
        )
        self.hold_joint_idx = np.array(
            [i for i in range(len(ALL_JOINTS)) if i not in set(self.action_joint_idx.tolist())],
            dtype=int,
        )
        self.n_action = len(self.action_joint_idx)
        # per-joint delta scale (gripper may close faster; see config)
        self.action_scales = np.full(self.n_action, self.cfg.action_scale)
        if self.cfg.gripper_action_scale is not None and "gripper" in self.cfg.action_joints:
            self.action_scales[list(self.cfg.action_joints).index("gripper")] = (
                self.cfg.gripper_action_scale
            )

        self._nominal = {
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "dof_frictionloss": self.model.dof_frictionloss.copy(),
            "dof_armature": self.model.dof_armature.copy(),
            "actuator_gainprm": self.model.actuator_gainprm.copy(),
            "actuator_biasprm": self.model.actuator_biasprm.copy(),
            "gravity": self.model.opt.gravity.copy(),
        }

        self._configure_collisions()
        self._setup_task()  # subclass resolves task-specific ids

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_action,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self._obs_dim(),), dtype=np.float32
        )

        self._step_count = 0
        self._success_count = 0
        self._last_action = np.zeros(self.n_action, dtype=np.float32)
        self._hold_targets = np.zeros(len(self.hold_joint_idx), dtype=np.float64)
        self._renderer = None

    # -- model loading --------------------------------------------------------
    def _load_model(self, path: str):
        """Compile the MJCF, applying the jaw collision fix (see module docs)."""
        mujoco = self._mj
        model = mujoco.MjModel.from_xml_path(path)
        if not self.cfg.fix_gripper_collision:
            return model
        if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) < 0
               for b in _JAW_COLLISION_FIX):
            return model  # scene without the SO-101 gripper
        spec = mujoco.MjSpec.from_file(path)
        for body_name, (mesh_name, axis, blade_lim) in _JAW_COLLISION_FIX.items():
            body = spec.body(body_name)
            for g in body.geoms:
                if g.meshname == mesh_name and g.contype != 0:
                    g.contype = 0
                    g.conaffinity = 0
            for i, (center, half) in enumerate(
                self._jaw_pad_boxes(model, body_name, mesh_name, axis, blade_lim)
            ):
                pad = body.add_geom(
                    name=f"{body_name}_pad{i}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=center.tolist(),
                    size=half.tolist(),
                    group=3,
                )
                pad.contype = 1
                pad.conaffinity = 1
        return spec.compile()

    def _jaw_pad_boxes(self, model, body_name, mesh_name, axis, blade_lim,
                       n_slabs: int = 3):
        """Boxes (centre, half-size in body frame) covering a jaw's collision mesh."""
        mujoco = self._mj
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        gid = next(
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid
            and model.geom_contype[g] != 0
            and model.geom_dataid[g] >= 0
            and mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]
            ) == mesh_name
        )
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
        verts = model.geom_pos[gid] + model.mesh_vert[va:va + vn] @ rot.reshape(3, 3).T

        def bbox(points, lo=None, hi=None):
            mins, maxs = points.min(axis=0), points.max(axis=0)
            if lo is not None:
                mins[axis], maxs[axis] = lo, hi
            return (mins + maxs) / 2, np.maximum((maxs - mins) / 2, 0.002)

        blade = verts[verts[:, axis] < blade_lim]
        edges = np.linspace(blade[:, axis].min(), blade_lim, n_slabs + 1)
        boxes = [
            bbox(blade[(blade[:, axis] >= lo) & (blade[:, axis] <= hi)], lo, hi)
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        boxes.append(bbox(verts[verts[:, axis] >= blade_lim]))  # palm / hinge base
        return boxes

    # -- id helpers ---------------------------------------------------------
    def _jid(self, n): return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, n)
    def _aid(self, n): return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_ACTUATOR, n)
    def _sid(self, n): return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SITE, n)
    def _bid(self, n): return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, n)

    def _body_has_free_joint(self, body_id: int) -> bool:
        jadr = self.model.body_jntadr[body_id]
        jnum = self.model.body_jntnum[body_id]
        for j in range(jadr, jadr + jnum):
            if self.model.jnt_type[j] == self._mj.mjtJoint.mjJNT_FREE:
                return True
        return False

    # -- collisions ---------------------------------------------------------
    def _configure_collisions(self) -> None:
        """3-class contype/conaffinity scheme (unless self_collision=True):

          floor/world : contype=1, conaffinity=6   (collides with arm+object)
          object      : contype=4, conaffinity=3   (collides with floor+arm)
          arm         : contype=2, conaffinity=5   (collides with floor+object, not itself)

        So the arm never self-collides but still hits the floor and can grasp
        objects. Pure-visual geoms (contype==conaffinity==0, e.g. markers and the
        cosmetic "water") are left untouched. Applied in-code so the vendored
        MJCF stays pristine.
        """
        if self.cfg.self_collision:
            return
        for g in range(self.model.ngeom):
            if self.model.geom_contype[g] == 0 and self.model.geom_conaffinity[g] == 0:
                continue  # visual-only geom
            body = self.model.geom_bodyid[g]
            if self._body_has_free_joint(body):
                ct, ca = 4, 3        # object (cup)
            elif body == 0:
                ct, ca = 1, 6        # world / floor
            else:
                ct, ca = 2, 5        # arm link
            self.model.geom_contype[g] = ct
            self.model.geom_conaffinity[g] = ca

    # -- domain randomization ----------------------------------------------
    def _apply_domain_randomization(self) -> None:
        n = self._nominal
        m = self.model
        if not self.dr.enabled:
            for k in ("body_mass", "body_inertia", "dof_damping", "dof_frictionloss",
                      "dof_armature", "actuator_gainprm", "actuator_biasprm"):
                getattr(m, k)[:] = n[k]
            m.opt.gravity[:] = n["gravity"]
            return

        rng, dr = self.np_random, self.dr
        nb, nd, nu = m.nbody, m.nv, m.nu
        m.body_mass[:] = n["body_mass"] * rng.uniform(*dr.mass_scale, nb)
        m.body_inertia[:] = n["body_inertia"] * rng.uniform(*dr.inertia_scale, (nb, 3))
        m.dof_damping[:] = n["dof_damping"] * rng.uniform(*dr.damping_scale, nd)
        m.dof_frictionloss[:] = n["dof_frictionloss"] * rng.uniform(*dr.frictionloss_scale, nd)
        m.dof_armature[:] = n["dof_armature"] * rng.uniform(*dr.armature_scale, nd)

        gain = n["actuator_gainprm"].copy()
        bias = n["actuator_biasprm"].copy()
        kp_f = rng.uniform(*dr.gain_scale, nu)
        kv_f = rng.uniform(*dr.kv_scale, nu)
        gain[:, 0] *= kp_f
        bias[:, 1] *= kp_f
        bias[:, 2] *= kv_f
        m.actuator_gainprm[:] = gain
        m.actuator_biasprm[:] = bias

        m.opt.gravity[:] = n["gravity"]
        m.opt.gravity[2] += rng.normal(0.0, dr.gravity_std)

    # -- control ------------------------------------------------------------
    def _apply_action(self, action: np.ndarray) -> None:
        cur = self.data.qpos[self._qadr[self.action_joint_idx]]
        lo = self._jnt_range[self.action_joint_idx, 0]
        hi = self._jnt_range[self.action_joint_idx, 1]
        if self.cfg.action_mode == "delta":
            target = cur + action * self.action_scales
        elif self.cfg.action_mode == "absolute":
            target = lo + (action + 1.0) * 0.5 * (hi - lo)
        else:
            raise ValueError(f"unknown action_mode {self.cfg.action_mode!r}")
        target = np.clip(target, lo, hi)
        self.data.ctrl[self._act_ids[self.action_joint_idx]] = target
        if len(self.hold_joint_idx):
            self.data.ctrl[self._act_ids[self.hold_joint_idx]] = self._hold_targets
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_domain_randomization()
        self._mj.mj_resetData(self.model, self.data)

        home = (np.zeros(len(ALL_JOINTS)) if self.cfg.init_qpos is None
                else np.asarray(self.cfg.init_qpos, dtype=float))
        qpos = home + self.np_random.uniform(
            -self.cfg.init_qpos_noise, self.cfg.init_qpos_noise, size=len(ALL_JOINTS)
        )
        qpos[ALL_JOINTS.index("gripper")] = self.cfg.gripper_hold  # start open
        qpos = np.clip(qpos, self._jnt_range[:, 0], self._jnt_range[:, 1])
        self.data.qpos[self._qadr] = qpos
        self.data.qvel[self._vadr] = 0.0
        self.data.ctrl[self._act_ids] = qpos
        self._hold_targets = qpos[self.hold_joint_idx].copy()

        self._reset_task()
        self._mj.mj_forward(self.model, self.data)

        self._step_count = 0
        self._success_count = 0
        self._last_action = np.zeros(self.n_action, dtype=np.float32)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._apply_action(action)
        self._step_count += 1
        self._last_action = action.astype(np.float32)

        obs = self._get_obs()
        # Divergence guard: an under-trained policy can drive the arm into the
        # floor and the contact launches it (positions/velocities blow up). End
        # the episode with a penalty instead of logging garbage — and never feed
        # NaN/inf observations to the trainer (that poisons VecNormalize).
        # Finiteness is checked over the whole state (a NaN anywhere — e.g. a
        # flung free object — would poison the observation), but the speed
        # threshold looks only at the ARM dofs: a fast-moving cup is bad play,
        # not a numerical blow-up, and shouldn't abort the episode.
        arm_speed = float(np.max(np.abs(self.data.qvel[self._vadr])))
        diverged = (
            not np.all(np.isfinite(self.data.qpos))
            or not np.all(np.isfinite(self.data.qvel))
            or arm_speed > self.cfg.max_joint_speed
        )
        if diverged:
            obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0),
                          -50.0, 50.0).astype(np.float32)
            return obs, -10.0, True, False, {"is_success": False, "unstable": True, "dist": 99.9}

        reward, terminated, info = self._reward_and_done(action)
        truncated = self._step_count >= self.cfg.max_episode_steps
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -- proprio helper shared by obs builders ------------------------------
    def _proprio(self):
        qpos = self.data.qpos[self._qadr].copy()
        qvel = self.data.qvel[self._vadr].copy()
        if self.cfg.obs_joint_noise_std > 0:
            qpos = qpos + self.np_random.normal(0, self.cfg.obs_joint_noise_std, qpos.shape)
        if self.cfg.obs_vel_noise_std > 0:
            qvel = qvel + self.np_random.normal(0, self.cfg.obs_vel_noise_std, qvel.shape)
        return qpos, qvel

    # -- rendering ----------------------------------------------------------
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- task hooks (subclasses implement) ----------------------------------
    def _setup_task(self) -> None: raise NotImplementedError
    def _obs_dim(self) -> int: raise NotImplementedError
    def _get_obs(self) -> np.ndarray: raise NotImplementedError
    def _get_info(self) -> dict: raise NotImplementedError
    def _reset_task(self) -> None: raise NotImplementedError
    def _reward_and_done(self, action) -> tuple[float, bool, dict]: raise NotImplementedError
