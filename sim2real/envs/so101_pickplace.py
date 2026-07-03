"""SO-101 pick-and-place task (MuJoCo + Gymnasium).

Grasp a free-standing cup and place it on a goal pad. The gripper is part of the
action space (6 controlled joints), and the cup is a rigid free body.

Reward design notes (hard-won — read before touching):

* Dense terms are POSITIVE shaped bonuses ``1 - tanh(d/scale)``. An earlier
  all-negative dense reward made early termination profitable: flinging the cup
  out of the workspace (-5, episode over) beat playing on at ~-0.15/step, and
  PPO learned exactly that (62% of episodes ended cup-lost, zero grasps).
* Jaw-pad contact shaping (``w_contact`` per jaw touching the cup) bridges the
  exploration gap between "TCP near cup" and "cup lifted" — without it the
  policy has to discover enclose+close+lift by pure chance.
* ``placed`` requires the cup to have been LIFTED at some point
  (``min_lift_for_success``); otherwise shoving the cup onto the pad counts.
* **The cup holds fluid.** Tilting it past ``spill_tilt`` spills — episode
  over, penalty, no recovery (MuJoCo has no fluid; the tilt limit models the
  spill). Dense ``w_upright`` shaping keeps gradient before that cliff, and
  ``placed`` additionally requires the cup upright (``place_tilt``). Yaw spin
  about the vertical axis is allowed — it doesn't spill.
* Success does NOT terminate the episode: the placed state keeps paying
  ``success_bonus`` per step, so placing early maximizes return. Terminating
  would cut the payment off and teach the policy to avoid placing.

**Sim2real caveat:** the observation includes the cup pose, which the physical
arm cannot measure from its servos alone. Real deployment needs object
perception (an overhead camera / AprilTag) feeding the cup pose into the same
observation slot. Everything else transfers as in the reach task.
"""

from __future__ import annotations

import numpy as np

from sim2real.config import ALL_JOINTS
from sim2real.envs.base import SO101MujocoBase


class SO101PickPlaceEnv(SO101MujocoBase):
    """Pick up the cup and place it on the goal pad."""

    def _setup_task(self) -> None:
        self._tcp_site_id = self._sid("gripperframe")
        self._cup_bid = self._bid("cup")
        cup_jid = int(self.model.body_jntadr[self._cup_bid])
        cadr = int(self.model.jnt_qposadr[cup_jid])
        vadr = int(self.model.jnt_dofadr[cup_jid])
        self._cup_qadr = np.arange(cadr, cadr + 7)   # 3 pos + 4 quat
        self._cup_vadr = np.arange(vadr, vadr + 6)    # 6 dof
        self._goal_bid = self._bid("goal")
        self._goal_mocap_id = int(self.model.body_mocapid[self._goal_bid])
        if self._goal_mocap_id < 0:
            raise RuntimeError("`goal` body must be a mocap body in the scene XML")
        # red start pad (visual marker under the cup's spawn; optional)
        start_bid = self._bid("start")
        self._start_mocap_id = (int(self.model.body_mocapid[start_bid])
                                if start_bid >= 0 else -1)

        # jaw-pad geom ids for grasp detection (added by the base-class collision
        # fix; grasping is impossible against the raw convex-hulled jaw meshes)
        mj = self._mj
        self._cup_gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "cup_glass")
        self._fixed_pads = self._pad_gids("gripper")
        self._moving_pads = self._pad_gids("moving_jaw_so101_v1")
        if not self._fixed_pads or not self._moving_pads:
            raise RuntimeError(
                "jaw pad geoms not found — SO101PickPlace needs "
                "EnvConfig.fix_gripper_collision=True (hulled jaw meshes cannot grasp)"
            )
        self._max_lift = 0.0
        self._ever_placed = False

    def _pad_gids(self, body_name: str) -> frozenset[int]:
        mj = self._mj
        gids = []
        for g in range(self.model.ngeom):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith(f"{body_name}_pad"):
                gids.append(g)
        return frozenset(gids)

    def _obs_dim(self) -> int:
        # qpos(6)+qvel(6)+tcp(3)+cup(3)+(cup-tcp)(3)+goal(3)+(cup-goal)(3)
        dim = len(ALL_JOINTS) * 2 + 3 * 5
        if self.cfg.include_last_action:
            dim += self.n_action
        return dim

    # -- task quantities ----------------------------------------------------
    def _tcp(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site_id].copy()

    def _cup(self) -> np.ndarray:
        return self.data.xpos[self._cup_bid].copy()

    def _goal(self) -> np.ndarray:
        return self.data.mocap_pos[self._goal_mocap_id].copy()

    def _cup_speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self._cup_vadr[:3]]))

    def _cup_cos_tilt(self) -> float:
        """cos(tilt) of the cup axis vs vertical (1 = upright, 0 = on its side)."""
        return float(self.data.xmat[self._cup_bid].reshape(3, 3)[2, 2])

    def _jaw_touches(self) -> tuple[bool, bool]:
        """(fixed jaw touching cup, moving jaw touching cup)."""
        touch_fixed = touch_moving = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._cup_gid:
                other = c.geom2
            elif c.geom2 == self._cup_gid:
                other = c.geom1
            else:
                continue
            if other in self._fixed_pads:
                touch_fixed = True
            elif other in self._moving_pads:
                touch_moving = True
        return touch_fixed, touch_moving

    # Fingers-down pre-grasp arm pose with the jaw mouth at (0.24, 0) on the
    # floor (found by IK; regression-tested via test_gripper_can_hold_cup).
    # shoulder_pan rotates the whole arm about z, sweeping the mouth along the
    # r=0.24 arc, so one pose parameterizes a family of curriculum starts.
    _PREGRASP_QPOS = np.array([-0.0009, 0.2208, 0.1735, 1.1757, 0.0209])
    _PREGRASP_GRIPPER = 1.2

    def _reset_task(self) -> None:
        rng, c = self.np_random, self.cfg
        u = rng.random()
        place_start = carry_start = False
        p_place = c.place_init_prob
        p_carry = p_place + c.carry_init_prob
        p_hold = p_carry + c.hold_init_prob
        p_grasp = p_hold + c.grasp_init_prob
        p_appr = p_grasp + c.approach_init_prob
        if u < p_place:
            cup_xy = self._reset_in_mouth(closed=True, elevated=True)
            place_start = True
        elif u < p_carry:
            # transport rung: held aloft, goal sampled far away as usual
            cup_xy = self._reset_in_mouth(closed=True, elevated=True)
            carry_start = True
        elif u < p_hold:
            cup_xy = self._reset_in_mouth(closed=True)
        elif u < p_grasp:
            cup_xy = self._reset_in_mouth(closed=False)
        elif u < p_appr:
            # final-approach rung: half the starts are lateral (cup a few cm
            # beside the open jaws), half from above (gripper raised over a
            # nearby cup — descending tips the cup far less than sweeping
            # sideways into its rim, which is how most approach spills happen)
            if rng.random() < 0.5:
                standoff = rng.uniform(0.15, 0.5) * rng.choice([-1.0, 1.0])
                cup_xy = self._reset_in_mouth(closed=False, standoff=standoff)
            else:
                standoff = rng.uniform(0.0, 0.2) * rng.choice([-1.0, 1.0])
                cup_xy = self._reset_in_mouth(closed=False, elevated=True,
                                              standoff=standoff)
        else:
            cup_xy = rng.uniform(c.cup_init_low, c.cup_init_high)
            if c.opposite_sides:
                side = rng.choice([-1.0, 1.0])
                cup_xy[1] = side * rng.uniform(c.side_min_y, abs(c.cup_init_high[1]))
            # cup upright on the floor, at rest
            self.data.qpos[self._cup_qadr] = [cup_xy[0], cup_xy[1], c.cup_rest_z,
                                              1, 0, 0, 0]
            self.data.qvel[self._cup_vadr] = 0.0
        if place_start:
            # endgame curriculum: the goal is a short hop from the held cup —
            # a mini-transport then set-down (never coincident with the cup;
            # the real task always has distinct, distant pads)
            ang = rng.uniform(0, 2 * np.pi)
            hop = rng.uniform(0.03, 0.07)
            goal_xy = np.clip(cup_xy + hop * np.array([np.cos(ang), np.sin(ang)]),
                              c.goal_low, c.goal_high)
            # count as lifted so a successful set-down registers as placed
            self._max_lift = c.min_lift_for_success
        elif carry_start:
            goal_xy = self._sample_goal(cup_xy)
            self._max_lift = c.min_lift_for_success  # already carried aloft
        else:
            goal_xy = self._sample_goal(cup_xy)
            self._max_lift = 0.0
        self.data.mocap_pos[self._goal_mocap_id] = [goal_xy[0], goal_xy[1], 0.001]
        if self._start_mocap_id >= 0:
            # red pad marks the cup's origin. For synthetic aloft starts
            # (place/carry) the cup never stood on the floor this episode, so
            # mark a plausible origin well away from the goal instead of the
            # cup's mid-air position.
            if place_start or carry_start:
                pad_xy = self._sample_goal(goal_xy)   # far from the goal
            else:
                pad_xy = cup_xy
            self.data.mocap_pos[self._start_mocap_id] = [pad_xy[0], pad_xy[1], 0.0008]
        self._ever_placed = False

    def _sample_goal(self, cup_xy: np.ndarray) -> np.ndarray:
        """Goal at least min_cup_goal_sep away; opposite side if configured."""
        rng, c = self.np_random, self.cfg
        for _ in range(20):
            goal_xy = rng.uniform(c.goal_low, c.goal_high)
            if c.opposite_sides:
                side = 1.0 if cup_xy[1] <= 0 else -1.0
                goal_xy[1] = side * rng.uniform(c.side_min_y, abs(c.goal_high[1]))
            if np.linalg.norm(goal_xy - cup_xy) >= c.min_cup_goal_sep:
                break
        return goal_xy

    def _reset_in_mouth(self, closed: bool, elevated: bool = False,
                        standoff: float = 0.0) -> np.ndarray:
        """Curriculum start: cup between the jaws (open, or closed = holding).

        Returns the cup xy. For the closed variant a short physics settle
        resolves the pad-cup contacts so the episode starts in a stable grip.
        ``elevated`` raises the grasp pose so the held cup hovers 2-7 cm up
        (used by the place curriculum). ``standoff`` (rad, open jaws only)
        rotates the cup away from the mouth along the workspace arc, so the
        episode starts with the cup a few cm BESIDE the gripper (approach
        curriculum).
        """
        rng = self.np_random
        qpos = self._PREGRASP_QPOS + rng.uniform(-0.03, 0.03, 5)
        qpos[0] += rng.uniform(-0.45, 0.45)          # sweep the r=0.24 arc
        if closed:
            gq = rng.uniform(0.42, 0.50)             # jaws at cup width
            g_ctrl = rng.uniform(0.05, 0.12)         # squeeze target
        else:
            gq = self._PREGRASP_GRIPPER + rng.uniform(-0.15, 0.3)
            g_ctrl = gq
        full = np.clip(np.concatenate([qpos, [gq]]),
                       self._jnt_range[:, 0], self._jnt_range[:, 1])
        self.data.qpos[self._qadr] = full
        self.data.qvel[self._vadr] = 0.0
        self.data.ctrl[self._act_ids] = full          # hold the pose, not home
        self.data.ctrl[self._act_ids[-1]] = g_ctrl
        self._mj.mj_forward(self.model, self.data)
        # cup goes mid-mouth: ~30 mm from the fixed-jaw tip toward the moving jaw
        grip_bid = self._bid("gripper")
        rot = self.data.xmat[grip_bid].reshape(3, 3)
        spread = 0.004 if closed else 0.010
        site = self.data.site_xpos[self._tcp_site_id]
        mouth = site + (0.030 + rng.uniform(-spread, spread)) * rot[:, 0]
        if standoff:
            cz, sz = np.cos(standoff), np.sin(standoff)
            mouth[:2] = [cz * mouth[0] - sz * mouth[1],
                         sz * mouth[0] + cz * mouth[1]]
        self.data.qpos[self._cup_qadr] = [mouth[0], mouth[1], self.cfg.cup_rest_z,
                                          1, 0, 0, 0]
        self.data.qvel[self._cup_vadr] = 0.0
        if closed:
            for _ in range(40):                       # settle into the grip
                self._mj.mj_step(self.model, self.data)
        if elevated:
            # form the grip on the floor first, then raise the shoulder ctrl
            # gradually (a squeezed cup teleported mid-air gets squirted out;
            # a formed grip survives the lift — same recipe as the probes)
            up = rng.uniform(0.10, 0.35)              # site +2..7 cm
            q_up = full[:5].copy()
            q_up[1] -= up
            q_up[3] += up                             # wrist keeps fingers down
            q_up = np.clip(q_up, self._jnt_range[:5, 0], self._jnt_range[:5, 1])
            for a in np.linspace(0.0, 1.0, 60):
                self.data.ctrl[self._act_ids[:5]] = (1 - a) * full[:5] + a * q_up
                self._mj.mj_step(self.model, self.data)
            return self.data.xpos[self._cup_bid][:2].copy()
        return mouth[:2].copy()

    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self._proprio()
        tcp, cup, goal = self._tcp(), self._cup(), self._goal()
        parts = [qpos, qvel, tcp, cup, cup - tcp, goal, cup - goal]
        if self.cfg.include_last_action:
            parts.append(self._last_action)
        return np.concatenate(parts).astype(np.float32)

    # -- reward -------------------------------------------------------------
    def _metrics(self):
        c = self.cfg
        tcp, cup, goal = self._tcp(), self._cup(), self._goal()
        d_tcp_cup = float(np.linalg.norm(tcp - cup))
        d_cup_goal = float(np.linalg.norm(cup[:2] - goal[:2]))
        lift = float(cup[2] - c.cup_rest_z)
        d_place = float(np.linalg.norm(cup - [goal[0], goal[1], c.cup_rest_z]))
        touch_fixed, touch_moving = self._jaw_touches()
        n_touch = int(touch_fixed) + int(touch_moving)
        # grasped + (lifted OR in the landing zone). Lift-only gating is a
        # reward cliff during the final cm of set-down; contact-only gating
        # pays for dragging the cup along the floor. This is the middle path.
        holding = n_touch == 2 and (lift > c.lift_thresh
                                    or d_place < c.place_free_radius)
        cos_tilt = self._cup_cos_tilt()
        placed = (
            d_cup_goal < c.success_threshold
            and abs(lift) < c.place_height_tol
            and self._cup_speed() < c.cup_speed_thresh
            and self._max_lift >= c.min_lift_for_success
            and cos_tilt > np.cos(c.place_tilt)   # upright — it's a cup of fluid
        )
        return d_tcp_cup, d_cup_goal, d_place, lift, n_touch, cos_tilt, holding, placed

    def _reward_and_done(self, action):
        c = self.cfg
        # cup knocked out of the workspace (flung away or dropped) -> episode
        # over. With positive dense rewards this forfeits future return, so the
        # penalty genuinely bites (with negative rewards it was an exit reward).
        cup = self._cup()
        if float(np.linalg.norm(cup[:2])) > 0.55 or cup[2] > 0.5 or cup[2] < -0.05:
            return -5.0, True, {**self._get_info(False), "cup_lost": True}
        # the cup holds fluid: tilting past spill_tilt spills it — no recovery
        if self._cup_cos_tilt() < np.cos(c.spill_tilt):
            return float(c.spill_penalty), True, {**self._get_info(False), "spilled": True}

        self._max_lift = max(self._max_lift, float(cup[2] - c.cup_rest_z))
        (d_tcp_cup, d_cup_goal, d_place, lift, n_touch, cos_tilt,
         holding, placed) = self._metrics()
        qvel_arm = self.data.qvel[self._vadr]

        reward = c.w_reach * (1.0 - np.tanh(d_tcp_cup / c.reach_scale))
        reward += c.w_contact * n_touch
        reward += -c.w_upright * (1.0 - cos_tilt)   # keep the fluid level
        # lift bonus fades out near the goal so descending to place is free
        lift_gate = min(d_cup_goal / c.lift_fade_dist, 1.0)
        reward += lift_gate * c.w_lift * min(max(lift, 0.0), c.lift_target) / c.lift_target
        if holding:
            reward += c.w_grasp
            # 3D distance to the place point: descending over the pad pays.
            # Two length scales: the coarse term keeps a live gradient even
            # when the cup is held high/far (a single tanh(d/0.1) saturates
            # flat beyond ~25 cm and the policy feels no pull), the sharp term
            # gives precision at the pad.
            reward += c.w_transport * 0.5 * (
                (1.0 - np.tanh(d_place / (3.0 * c.transport_scale)))
                + (1.0 - np.tanh(d_place / c.transport_scale))
            )
        reward += -c.w_ctrl * float(np.sum(action**2))
        reward += -c.w_vel * float(np.sum(qvel_arm**2))
        if placed:
            reward += c.success_bonus

        self._success_count = self._success_count + 1 if placed else 0
        if self._success_count >= c.success_hold_steps:
            self._ever_placed = True  # sticky: episode counts as a success
        return float(reward), False, self._get_info(placed)

    def _get_info(self, placed: bool | None = None) -> dict:
        (d_tcp_cup, d_cup_goal, _, lift, n_touch, cos_tilt,
         holding, placed_now) = self._metrics()
        if placed is None:
            placed = placed_now
        return {
            "dist": d_cup_goal,           # cup-to-goal (primary task metric)
            "tcp_to_cup": d_tcp_cup,
            "lift": lift,
            "max_lift": self._max_lift,
            "n_touch": n_touch,
            "tilt": float(np.arccos(np.clip(cos_tilt, -1.0, 1.0))),
            "holding": bool(holding),
            "is_success": bool(self._ever_placed or placed),
            "cup": self._cup(),
            "goal": self._goal(),
        }
