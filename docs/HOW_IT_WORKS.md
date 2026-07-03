# How this was built — a sim2real learning guide

A walk-through of the whole SO-101 pipeline, written to *teach*: not just what
each file does, but the reasoning behind it and the concepts underneath. Read it
top-to-bottom once; after that each section stands alone. Code pointers use
`file: thing` (line numbers drift, names don't).

**Prereqs to get the most out of it:** basic Python, a rough idea of
reinforcement learning (agent, observation, action, reward), and the willingness
to run the little snippets as you go (`conda activate sim2real` first).

---

## Contents
0. [The one big idea](#0-the-one-big-idea)
1. [Anatomy of the pipeline](#1-anatomy-of-the-pipeline)
2. [Building the scene (MuJoCo / MJCF)](#2-building-the-scene)
3. [Making the arm "aware": observations](#3-making-the-arm-aware-observations)
4. [Moving the arm: actions & control](#4-moving-the-arm-actions--control)
5. [Telling it what "good" means: reward](#5-telling-it-what-good-means-reward)
6. [Crossing the reality gap: domain randomization](#6-crossing-the-reality-gap-domain-randomization)
7. [Training](#7-training)
8. [Seeing results](#8-seeing-results)
9. [Reaching the real arm](#9-reaching-the-real-arm)
10. [Extending it](#10-extending-it)
11. [Lessons the hard way](#11-lessons-the-hard-way)

---

## 0. The one big idea

Sim2real reinforcement learning is: **learn a policy in a fast simulator, then
run that exact policy on real hardware.** It works only if two things line up:

> **A simulated action must equal a real robot command, and a simulated
> observation must equal a real observation.**

Everything in this repo is organized to protect that equivalence. When you
understand a design choice, trace it back to this sentence and it will make
sense. Two concrete consequences you'll see everywhere:

- The policy outputs **joint position targets**, because that is literally what a
  Feetech STS3215 servo takes as a command (`goal_position`). A sim action *is* a
  servo command — no translation layer to get wrong.
- The observation is built from **things the real robot can measure** (joint
  angles/velocities, and forward-kinematics of those angles). If sim can see it
  but the real arm can't, it can't go in the observation.

The gap that remains — sim physics ≠ real physics — is bridged by **domain
randomization** (Section 6). That's the whole game in three bullets.

```
  ┌── SIMULATION (train here) ───────────────┐        ┌── REALITY (run here) ─────┐
  │  MuJoCo model of the SO-101              │        │  physical SO-101          │
  │  observation ── policy ── action(target) │  ≈≈≈▶  │  obs ── policy ── servos  │
  │  reward shapes the policy                │  same  │  (no reward, just run)    │
  │  domain randomization widens physics ────┼── ✔ ──▶│  real physics is "inside" │
  └──────────────────────────────────────────┘        └───────────────────────────┘
```

---

## 1. Anatomy of the pipeline

Data flows in one direction, from a robot model to a policy running on hardware:

```
 MJCF model ─▶ Gymnasium env ─▶ PPO (Stable-Baselines3) ─▶ trained policy
   (§2)          (§3,4,5)            (§7)                       │
                    ▲                                           ▼
        domain randomization (§6)              ONNX export (§9) ─▶ real arm (§9)
```

Where each piece lives:

| Concept | File | What to read it for |
|---|---|---|
| Robot model | `assets/so101/so101_new_calib.xml` | joints, actuators, the TCP site |
| Task scene | `assets/so101/reach_scene.xml`, `pickplace_scene.xml` | how a task is staged |
| Shared env engine | `sim2real/envs/base.py` | control, DR, collisions, the step/reset loop |
| Task definitions | `sim2real/envs/so101_reach.py`, `so101_pickplace.py` | obs + reward per task |
| Config | `sim2real/config.py` | every knob, one YAML == one run |
| Training | `sim2real/train.py`, `env_factory.py` | PPO wiring, vec envs, normalization |
| Kinematics | `sim2real/utils/kinematics.py` | TCP from joint angles (sim == real) |
| Deploy | `sim2real/deploy/` | servos, calibration, the real control loop |

The rest of this guide is those columns, in the order you'd think about them.

---

## 2. Building the scene

A MuJoCo simulation is defined by an **MJCF** file — XML describing bodies,
joints, geometry, and actuators. You don't hand-write the robot; you get an
accurate model and stage a task around it.

### 2.1 Get the robot, don't model it

Modeling a 3D arm by hand is error-prone. The SO-101 already has an official,
*calibrated* MuJoCo model (link lengths, masses, inertias, realistic servo
gains) in `TheRobotStudio/SO-ARM100`. We **vendored** it — copied it into
`assets/so101/` unchanged — so the repo is self-contained and reproducible
(`scripts/fetch_assets.sh` re-syncs it). Rule of thumb: keep third-party assets
pristine and never hand-edit them; layer your changes in your own files.

### 2.2 How to read an MJCF

Open `assets/so101/so101_new_calib.xml`. The pieces that matter:

- **`<body>`** — a rigid link. Bodies nest to form the kinematic chain:
  `base → shoulder → upper_arm → lower_arm → wrist → gripper → moving_jaw`.
- **`<joint>`** — how a body moves relative to its parent. All six SO-101 joints
  are `hinge` (1 rotational DoF) with a `range` (limits, in radians).
- **`<inertial>`** — mass + inertia of the link. This is the *physics* of the
  arm; domain randomization perturbs exactly these numbers.
- **`<geom>`** — collision/visual shape (here, meshes). Visual geoms are group 2,
  collision geoms group 3.
- **`<actuator>` → `<position ...>`** — a **position servo**: you command a
  target angle (`ctrl`), and it drives the joint there with gains `kp`, `kv`.
  This is the single most important line for sim2real (Section 4).
- **`<site name="gripperframe">`** — a named coordinate frame at the fingertip.
  We read its world position as the **TCP** (tool-centre-point).

The SO-101 is 6 actuated joints: `shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper`. That fixed order lives in
`config.ALL_JOINTS` and everything indexes by it.

### 2.3 Stage a task around the robot

A **task scene** includes the robot and adds the world and the task props. Look
at `assets/so101/reach_scene.xml`:

```xml
<mujoco model="so101_reach">
  <include file="so101_new_calib.xml"/>     <!-- the robot, untouched -->
  ...
  <worldbody>
    <geom name="floor" type="plane" .../>   <!-- the table/ground -->
    <body name="target" mocap="true" ...>   <!-- the reach target -->
      <geom type="sphere" size="0.015" rgba="0.9 0.1 0.1 0.55" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
```

Two techniques worth internalizing:

- **`mocap="true"` bodies** are *kinematic* — they have no physics, and you
  teleport them by writing `data.mocap_pos[id]`. Perfect for a "target" or a
  "goal pad" you reposition every episode. They don't add to the state vector.
- **Free bodies** are the opposite: a `<freejoint/>` gives a body 6 DoF so it
  falls, gets pushed, and can be grasped. The cup in `pickplace_scene.xml` is a
  free body — that's what makes pick-and-place a *physics* task, not a scripted
  one. A free joint adds 7 numbers to `qpos` (x,y,z + quaternion) and 6 to
  `qvel`.

The cup also shows the "fake fluid" trick: a translucent solid cylinder (the
glass, collidable) plus a shorter opaque cylinder (the "water", `contype=0
conaffinity=0` so it's visual-only). MuJoCo has **no liquid simulation**; the
water is cosmetic and rides rigidly with the cup.

### 2.4 Inspect a model — the single most useful habit

You never guess names or indices; you query the compiled model. This is the exact
snippet used while building the env:

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("assets/so101/reach_scene.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)                      # compute positions from the state
print("nq nv nu:", m.nq, m.nv, m.nu)         # #position, #velocity, #actuators
print([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
print("TCP:", d.site_xpos[sid])              # world XYZ of the fingertip
```

Key MuJoCo objects: **`MjModel`** is constant (the description); **`MjData`** is
the changing state (`qpos`, `qvel`, `ctrl`, `xpos`, `site_xpos`, ...).
`mj_step` advances physics; `mj_forward` just recomputes derived quantities
(positions, sites) without integrating time. Because joints can appear in any
order and objects add DoF, we **index joints by name**, never by hard-coded
slice — see `base.py: SO101MujocoBase.__init__` building `_qadr`/`_vadr` from
`jnt_qposadr`/`jnt_dofadr`.

### 2.5 Collisions, on purpose

Meshes exported from CAD often overlap slightly at the joints, which makes the
solver fight itself. `base.py: _configure_collisions` fixes this **in code** (so
the vendored file stays pristine) with a 3-class scheme using MuJoCo's
`contype`/`conaffinity` bitmasks: two geoms collide iff
`(contype_A & conaffinity_B) | (contype_B & conaffinity_A) != 0`.

```
floor  : contype=1  conaffinity=6      # collides with arm + objects
arm    : contype=2  conaffinity=5      # collides with floor + objects, NOT itself
object : contype=4  conaffinity=3      # collides with floor + arm
```

So the arm can hit the floor and grasp the cup, but never spuriously collides
with its own links. Set `EnvConfig.self_collision=True` to restore full contacts.

**The convex-hull trap (read this before adding any graspable object).** MuJoCo
collides mesh geoms as their **convex hulls**. The vendored SO-101 model reuses
the concave visual meshes as collision geoms, and the hulls of the palm+fixed-jaw
piece and the moving jaw *fill the space between the jaws*: in collision space
the gripper mouth is solid, and no object can ever be grasped — it gets expelled
with deep soft penetration. This is invisible in the viewer (you see the visual
meshes) and produced a 5M-step training run with literally zero grasps.
`base.py: _load_model` fixes it at load time via `MjSpec`
(`EnvConfig.fix_gripper_collision`, on by default): collision is disabled on the
two jaw meshes and replaced with box "pads" fitted to the mesh vertices
(`gripper_pad*`, `moving_jaw_so101_v1_pad*`). The pads double as the grasp
detector for the pick-place reward, and `test_gripper_can_hold_cup` guards the
whole arrangement.

---

## 3. Making the arm "aware": observations

"Awareness" is the **observation** — the fixed-length vector the policy sees each
step. Designing it is half the problem. Two rules:

1. Include what the policy needs to make the decision.
2. Include **only** what the real robot can also produce (the sim2real line).

### 3.1 What the reach policy sees

`so101_reach.py: _get_obs` concatenates (26 numbers):

```
[ joint angles (6), joint velocities (6),      # proprioception (from servos)
  TCP xyz (3),                                  # fingertip position (forward kinematics)
  target xyz (3), target - TCP (3),             # the goal, and the vector to it
  last action (5) ]                             # what it just commanded (helps smoothness)
```

Every component is measurable on the real arm: the STS3215 servos report angle
and speed; the TCP is *computed* from the angles (next section); the target is
defined by the task. The `target - TCP` vector is redundant with the two
positions, but handing the network the difference directly makes learning easier
— a cheap, legitimate trick.

### 3.2 Proprioception vs perception — the sim2real line

Reach uses **proprioception only** (the robot sensing itself), so it transfers to
hardware with nothing extra. Pick-and-place (`so101_pickplace.py: _get_obs`)
adds the **cup pose**:

```
[ ...proprio..., TCP, cup xyz, cup - TCP, goal xyz, cup - goal, last action ]  # 33
```

The cup pose is *not* something the servos can measure. In sim we read it for
free from `data.xpos[cup_body]`; on the real arm you'd need **perception** — an
overhead camera or an AprilTag — to fill that slot. This is the honest cost of
manipulation, and it's called out wherever it matters. Recognizing which
observation components are "free in sim but expensive in reality" is a core
sim2real skill.

### 3.3 Forward kinematics: TCP from joint angles

The policy wants the fingertip position, but the servos only give angles.
**Forward kinematics (FK)** maps joint angles → end-effector pose. Rather than
derive equations (and risk sim ≠ real), we reuse the *same MuJoCo model* as a
calculator — `utils/kinematics.py: ForwardKinematics`:

```python
self.data.qpos[self._qadr] = joint_angles
mujoco.mj_kinematics(self.model, self.data)      # just kinematics, no dynamics
return self.data.site_xpos[self.tcp_site_id]     # TCP in the base frame
```

One implementation, used by both the simulator and the real-robot controller.
That "single source of truth" is why the FK gap between sim and real is zero
(the tests check it to 1e-5 m).

### 3.4 Normalization (and why it must travel with the policy)

Neural nets learn best when inputs are ~zero-mean, unit-variance. During training
`VecNormalize` tracks a running mean/std of observations and normalizes them.
**The catch:** the policy was trained on *normalized* obs, so at deploy time you
must normalize identically — with the *same* stats. Forgetting this silently
breaks the transferred policy. Our fix (Section 9): bake the stats into the
exported policy so raw observations go in and it's impossible to desync.

---

## 4. Moving the arm: actions & control

### 4.1 Actions are position targets

The action space is `Box(-1, 1, shape=(n_joints,))` — a normalized number per
controlled joint. `base.py: _apply_action` turns it into a joint-angle target and
writes it to `data.ctrl`, which the MJCF's **position actuators** track. On the
real arm the very same target becomes the servo's `goal_position`. That's the
action-equivalence half of the big idea, made concrete.

Reach controls the 5 arm joints (gripper held open); pick-and-place adds the
gripper (6). Which joints are actuated is just `EnvConfig.action_joints`; the
rest are held in place. No env code changes between the two — only config.

### 4.2 delta vs absolute

```python
if action_mode == "delta":                       # default
    target = current_angle + action * action_scale   # small step each tick
else:  # "absolute"
    target = lo + (action + 1)/2 * (hi - lo)          # map [-1,1] across the range
```

**Delta** (relative) control is the sim2real-friendly default: each step nudges
the joints a little (`action_scale` radians), which keeps motion smooth and
bounded — real servos and gearboxes dislike large instantaneous jumps. Targets
are always clipped to joint limits.

### 4.3 Control rate and `frame_skip`

The simulator integrates at 500 Hz (2 ms), but a real control loop runs slower
(~25 Hz here). So each `env.step` holds the target for several physics steps:

```python
frame_skip = round((1 / control_freq) / sim_timestep)   # 500/25 = 20
for _ in range(frame_skip):
    mujoco.mj_step(model, data)
```

Matching the sim control rate to the real loop rate is a subtle but real part of
transfer — the policy learns dynamics at the cadence it will actually run at.

---

## 5. Telling it what "good" means: reward

Reward is how you *specify the task*. RL maximizes cumulative reward, so the
reward function is your real "programming interface" to behavior.

### 5.1 Reach: dense shaping

`so101_reach.py: _reach_reward`:

```python
reward = ( -w_dist * distance                       # closer is better (dense gradient)
           + w_near * exp(-distance / near_scale)    # sharp bonus very near the goal
           - w_ctrl * ||action||^2                   # penalize big/jerky commands
           - w_vel  * ||joint_velocity||^2 )         # penalize thrashing (smoothness)
if distance < success_threshold: reward += success_bonus
```

The `-distance` term gives a gradient *everywhere* (the agent always knows which
way is better), which is what makes reach easy to learn. The control/velocity
penalties aren't just tidiness — smooth policies transfer to hardware far better
than twitchy ones.

### 5.2 Pick-and-place: staged shaping

Grasping is a hard *exploration* problem: random actions almost never grasp, so a
pure "reward only on success" signal never fires. The fix is a **staged** reward
that pays for progress along the intended sequence — `so101_pickplace.py:
_reward_and_done`:

```
approach   : +w_reach   * (1 - tanh(distance(TCP, cup) / reach_scale))
touch      : +w_contact * (#jaw pads touching the cup)          # 0, 1 or 2
upright    : -w_upright * (1 - cos(cup tilt))    # it's a cup of FLUID
lift       : +w_lift    * clip(cup_height, 0, lift_target)      # fades near goal
hold       : +w_grasp        (both jaws touch + lifted, or in the landing zone)
transport  : +w_transport * two-scale tanh shaping toward the PLACE POINT
             (goal xy at rest height, 3D — so descending over the pad pays)
place      : +success_bonus per step        (cup at goal, on table, still,
                                             UPRIGHT, AND lifted earlier)
spill      : tilt > spill_tilt (~26°)  ->  -5 and episode over (no recovery)
```

MuJoCo has no fluid simulation — the "water" is a rigidly-attached visual — so
spilling is modelled as the tilt limit above. Yaw spin is allowed (it doesn't
spill); roll/pitch is what kills you.

Every dense term is a **positive bonus**, success does **not** end the episode,
and each of those choices was paid for in blood (see §5.3). "Holding" is
detected from *jaw-pad contact + lift* — the collision fix (§2.5) gives the jaws
named pad geoms, so both-jaws-touching is cheap to check and can't be faked by
pressing the cup from one side.

Even with shaping, from-scratch discovery of enclose→close→lift→carry→place is
too long a chain for PPO. Three **curriculum resets** (train-only; eval always
starts from scratch) hand the policy the later stages directly:
`grasp_init_prob` starts with open jaws around the cup, `hold_init_prob` starts
already holding it, `place_init_prob` starts holding it just above the goal.

### 5.3 How to think about reward design

- Prefer **dense** (distance-like) terms over sparse ones — they give gradient.
- **Shape toward the sequence** you want (approach → grasp → carry → place).
- Add small **regularizers** (action/velocity) for sim2real smoothness.
- Reward the *outcome you can measure on hardware*, and beware rewards the agent
  can "cheat". Real failures from this repo's history, all observed at 5M steps:
  - **All-negative dense reward + early termination = suicide.** When every step
    costs ~-0.15 and losing the cup ends the episode at -5, flinging the cup out
    of the workspace *maximizes return*. The policy learned exactly that (62% of
    episodes). Positive bonuses make survival valuable and termination a real
    penalty.
  - **Success that doesn't require a lift = shoving.** The early "placed" check
    only looked at cup-to-goal distance, so the policy pushed the cup onto the
    pad along the floor. `min_lift_for_success` closes the loophole.
  - **Terminating on success = avoiding success.** If the placed state pays per
    step, ending the episode on success cuts off that income and the policy
    learns to hover *near* success forever. Let it keep collecting.
  - **Saturated shaping = no gradient.** `tanh(d/0.1)` is flat beyond ~25 cm;
    a policy carrying the cup high felt no pull toward the pad. The two-scale
    term (`tanh(d/0.3) + tanh(d/0.1)`) keeps gradient alive everywhere.
  - **Gate income carefully: lift-gated transport is a cliff, ungated is a
    drag.** Requiring "cup lifted" for transport income cuts the reward off in
    the final 2 cm of set-down (policy refuses to place); requiring only jaw
    contact pays for *dragging* the cup along the floor (policy stops
    lifting). The fix: lifted **or** within `place_free_radius` of the place
    point — no income for dragging, no cliff at touchdown.
  - **Too much exploration noise breaks contact skills.** With SB3's default
    `log_std_init=0` (std≈1), no grip survives the behaviour policy, so grasp
    states never acquire value and the policy never learns to close. We train
    with `log_std_init=-1` and a small `ent_coef`.

---

## 6. Crossing the reality gap: domain randomization

No simulator is exactly reality. If you train on one fixed physics, the policy
overfits to that physics and fails on the real arm. **Domain randomization (DR)**
trains on a *distribution* of physics, so the real robot is just "one more
sample" the policy already handles. This is the single most important sim2real
technique.

### 6.1 Dynamics DR

Every reset, `base.py: _apply_domain_randomization` perturbs the physics:

```python
model.body_mass[:]      = nominal_mass      * U(0.8, 1.2)   # links 20% lighter/heavier
model.dof_damping[:]    = nominal_damping   * U(0.5, 1.6)   # joint friction/damping
model.dof_frictionloss  = ...               * U(0.5, 1.6)
model.actuator_gainprm  = nominal_kp        * U(0.8, 1.2)   # servo strength (kp/kv)
model.opt.gravity[2]   += Normal(0, 0.15)                   # even gravity wobbles
```

Two correctness points you should copy in your own work:

- **Cache the nominal values once** at construction and always scale *from
  nominal*. If you scaled the live values each reset, the randomization would
  **compound** and drift to zero/infinity. There's a regression test for exactly
  this (`test_dr_varies_but_does_not_compound`).
- **Scale servo gains consistently.** A MuJoCo position actuator stores `kp` in
  two places (`gainprm[0]` and `biasprm[1] = -kp`); scale both or you silently
  create an invalid actuator.

### 6.2 Command-path DR

Real commands don't reach the servos instantly or perfectly. Wrappers in
`sim2real/wrappers/` model that: `ActionLatencyWrapper` delays actions by a few
control steps (resampled per episode), `ActionNoiseWrapper` jitters them. Small,
but it teaches the policy to tolerate the real bus's timing.

### 6.3 Tuning DR

Wider DR = more robust but harder to learn and lower peak performance. The
workflow: train, evaluate **with** `--randomize` (that number predicts real-world
performance, not the clean one), and if the real arm underperforms, *widen DR and
retrain* — see `configs/reach_hard_dr.yaml`. When sim is great but reality isn't,
the answer is almost always "more DR," not "more code."

---

## 7. Training

We use **PPO** from Stable-Baselines3 — a solid, well-tested policy-gradient
algorithm good for continuous control. `train.py` + `env_factory.py` wire it up;
the concepts:

- **Vectorized envs** (`make_training_venv`): run N environments in parallel
  processes (`SubprocVecEnv`) so the GPU stays fed. More envs → more experience
  per wall-clock second. On the 4090, reach trains at ~1500+ fps.
- **`VecNormalize`**: normalizes observations (and reward) using running
  statistics — the stats we later bake into the ONNX.
- **Callbacks**: `CheckpointCallback` snapshots the model periodically; a custom
  `NormalizedEvalCallback` runs clean-physics evaluations *and first copies the
  normalization stats* from the trainer (a classic gotcha — eval with stale stats
  reports garbage success rates).
- **Everything is the config.** `train.py` just reads `configs/*.yaml` into the
  `Config` dataclass, so a run is fully described by one file (plus CLI
  overrides like `--timesteps`, `--n-envs`).

Run and watch:

```bash
make smoke                       # ~10 s: proves the plumbing end-to-end
make train                       # PPO on configs/reach.yaml (~2M steps)
make tb                          # TensorBoard: watch ep_rew_mean and success_rate climb
```

What to watch in TensorBoard: `rollout/ep_rew_mean` (should rise),
`eval/success_rate` (the real signal), and `train/explained_variance` (how well
the value function predicts returns; heading toward 1 is healthy).

---

## 8. Seeing results

Numbers tell you *if* it works; watching tells you *how*.

```bash
python -m sim2real.eval --run outputs/<run> --episodes 50            # success % + mm-to-target
python -m sim2real.eval --run outputs/<run> --randomize              # the transfer-predictive number
make watch RUN=outputs/<run>                                          # live 3D viewer
make video RUN=outputs/<run>                                          # headless mp4 (MUJOCO_GL=egl)
```

`eval.py` loads the model + normalization, runs deterministic rollouts, and
reports. `visualize.py` opens MuJoCo's interactive viewer stepping the policy in
real time (and `--video` renders offscreen). Always eval both nominal **and**
randomized — the gap between them tells you how much DR is costing you.

---

## 9. Reaching the real arm

Three steps take a trained policy to hardware.

**1. Export to ONNX with normalization baked in** (`export_policy.py`). We wrap
the trained network so it does `normalize(obs) → policy → action` *inside the
graph*:

```python
obs = (obs - mean) / sqrt(var + eps); obs = clip(obs, -clip, clip)   # baked in
action = policy(obs)
```

Now the robot feeds **raw** observations and can't desync the normalization. The
exporter verifies `ONNX(raw) == policy(normalized)` to ~1e-7 before saving. (Two
traps lived here: `model.predict` does *not* normalize, so the parity check must
compare against manually-normalized input; and Torch ≥2.9's default ONNX exporter
needs an extra package, so we force the proven `dynamo=False` path.)

**2. Calibrate the servos** (`docs/DEPLOYMENT.md`). The policy thinks in radians
about each joint's zero; the servos think in ticks (0–4095). Calibration is the
per-joint mapping (`center_ticks`, `joint_sign`) — `deploy/feetech_bus.py:
Calibration`:

```
radians = joint_sign * (tick - center_tick) * 2π / 4096
```

**3. Run the control loop** (`deploy/deploy_so101.py: PolicyController`). Each
tick it: reads servo angles → builds the *same* observation as training (TCP via
the shared FK) → runs the ONNX policy → maps the action to joint targets → clamps
for safety (max step, joint limits) → writes servo goal positions. The clever
part: a `SimPlantBus` implements the same bus interface backed by MuJoCo, so you
can run and debug the **entire** deploy path with zero hardware:

```bash
python -m sim2real.deploy.deploy_so101 --run outputs/<run> --target 0.25 0.05 0.2   # sim stand-in
python -m sim2real.deploy.deploy_so101 --run outputs/<run> --port /dev/ttyACM0 ...   # real arm
```

If the sim stand-in reaches the target but the real arm doesn't, the problem is
physics/DR, not your code — a very useful thing to be able to isolate.

---

## 10. Extending it

The env is a **template method**: `SO101MujocoBase` owns the machinery (model
loading, control, DR, collisions, the reset/step loop); a task subclass fills in
the hooks. To add a task (say, "stack two cups"):

1. Write a scene `assets/so101/stack_scene.xml` (`<include>` the robot, add
   objects/goals).
2. Subclass `SO101MujocoBase` and implement the five hooks: `_setup_task`
   (resolve ids), `_reset_task` (place objects), `_get_obs`, `_get_info`,
   `_reward_and_done`.
3. Register it in `sim2real/__init__.py` and add a `configs/stack.yaml` with the
   new `env_id`. Nothing else — training, eval, export, deploy already work
   because they only speak the config + Gym API.

Compare `so101_reach.py` (≈70 lines) and `so101_pickplace.py` (≈120 lines):
almost all of each file is task-specific obs/reward, because the shared 200 lines
live once in the base. That separation is the point.

---

## 11. Lessons the hard way

Real bugs hit while building this, kept here so you recognize them:

- **`from __future__ import annotations` turns dataclass field types into
  strings**, breaking any code that introspects them — resolve with
  `typing.get_type_hints` (`config.py`).
- **PyYAML can't dump tuples** — convert tuples→lists on save, restore on load.
- **`VecNormalize` stats must be synced to the eval env**, or success rates are
  meaningless.
- **`model.predict` does not apply `VecNormalize`** — matters when you verify a
  normalization-baked ONNX export.
- **Randomizing from live values compounds** — always randomize from cached
  nominal values.
- **MuJoCo offscreen render** needs `MUJOCO_GL=egl` and a big enough framebuffer
  (`<global offwidth/offheight>`); the **interactive** viewer needs `$DISPLAY`
  and `MUJOCO_GL` *unset*.
- **`conda run -n env pip …` can grab the system pip** — use `python -m pip`.

The meta-lesson: most sim2real effort is not the RL algorithm (that's a library
call). It's the plumbing that keeps sim and real speaking the same language —
observations, actions, kinematics, normalization, calibration. Get those right
and the learning mostly takes care of itself.

---

## Where to go next

- Run `make smoke`, then read `sim2real/envs/base.py` with this guide open — it's
  the spine of everything.
- Train reach to convergence (`make train`), `make watch` it, then eval
  `--randomize` to feel the DR cost.
- Read `CLAUDE.md` for the operating rules and the full gotcha list.
- **External:** MuJoCo docs (the "Overview" and "Modeling" chapters), the
  Stable-Baselines3 PPO guide, and the sim2real classic *"Sim-to-Real Transfer of
  Robotic Control with Dynamics Randomization"* (Peng et al.) for the theory
  behind Section 6.
