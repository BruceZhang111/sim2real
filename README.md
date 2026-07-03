# SO-101 Sim2Real

Reinforcement-learning **sim-to-real** pipeline for the low-cost
[SO-101 / SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) robotic arm
(Feetech STS3215 servos). Train a policy in MuJoCo with domain randomization,
then deploy it to the physical arm — the action interface and observations are
designed to transfer 1:1.

Two tasks ship today: **reach** (move the TCP to a target — fully sim2real from
proprioception) and **pick-and-place** (grasp a cup and place it on a goal pad —
adds the gripper to the action space; needs object perception on the real arm,
see [Tasks](#tasks)).

```
  MuJoCo (SO-101 MJCF)  ──►  PPO/SAC + domain randomization  ──►  ONNX policy  ──►  Feetech STS3215
        Gymnasium env            Stable-Baselines3            (normalization baked in)     real arm
```

- **Arm:** SO-101 / SO-ARM100, 5 arm joints + gripper (STS3215)
- **Sim:** MuJoCo 3, official calibrated MJCF vendored under `assets/so101/`
- **RL:** Stable-Baselines3 (PPO default, SAC available), GPU via PyTorch
- **Transfer:** dynamics + command-path domain randomization; position-target
  actions that map directly to servo goal positions; observations restricted to
  what the real servos + forward kinematics can provide

> 📚 **New to the codebase or to sim2real?** Read
> [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) — a from-first-principles build
> guide: the scene, observations ("awareness"), actions, reward, domain
> randomization, training, and deployment, with pointers into the code.

---

## Quickstart

```bash
# 1. environment (creates the `sim2real` conda env and installs the package)
make setup
conda activate sim2real

# 2. sanity-check the whole pipeline in ~10 s (tiny training run)
make smoke

# 3. train for real (PPO; edit the config to taste)
make train                            # reach:      python -m sim2real.train --config configs/reach.yaml
make train-pickplace                  # pick-place: python -m sim2real.train --config configs/pickplace.yaml
make tb                               # (optional) live TensorBoard curves while it trains

# 4. watch the trained policy in an interactive MuJoCo window
make watch RUN=outputs/<run>          # or: python -m sim2real.visualize --run outputs/<run>

# 5. evaluate: success rate + mm-to-target (add --randomize for the transfer proxy)
python -m sim2real.eval --run outputs/<run> --episodes 50
python -m sim2real.eval --run outputs/<run> --episodes 50 --randomize

# 6. export the policy to ONNX (normalization folded into the graph)
python -m sim2real.export_policy --run outputs/<run>

# 7. deploy — MuJoCo stand-in first (no hardware), then the real arm
python -m sim2real.deploy.deploy_so101 --run outputs/<run> --target 0.25 0.05 0.2
python -m sim2real.deploy.deploy_so101 --run outputs/<run> --port /dev/ttyACM0 --target 0.25 0.05 0.2
```

`<run>` is the folder created under `outputs/` (e.g. `ppo_reach_20260701_193000`).

---

## Visualize a run

Four ways to inspect a trained policy in `outputs/<run>/`:

| Command | What you get |
|---|---|
| `make watch RUN=outputs/<run>` | **Interactive** MuJoCo window — watch the policy live (reach target or cup+goal pad); orbit/zoom the camera. Needs a desktop display. |
| `make video RUN=outputs/<run>` | Renders `outputs/<run>/rollout.mp4` **headless** (works over SSH). |
| `make tb` | TensorBoard: reward, success rate, losses. |
| `python -m sim2real.eval --run outputs/<run>` | The numbers: success rate + distance metric. |

All of these read the task from the run's saved `config.yaml`, so the same
commands work for reach and pick-place runs. Add `--randomize` to `watch`/`eval`
to see behaviour under domain randomization. Useful `visualize` flags:
`--which final` (default `best`), `--video out.mp4`, `--episodes N`,
`--seed N` / `--episode N` (reproduce a specific scenario).

**Pick a specific episode.** By default `make watch` shows fresh random
scenarios. To make them reproducible and revisit one:

```bash
make watch RUN=outputs/<run> SEED=42     # reproducible sequence; each line prints its seed=
make watch RUN=outputs/<run> EP=7        # lock onto & replay just scenario 7
```

Each printed line shows `seed=<n>` — that number *is* the scenario's id, so note
an interesting one and pass it back as `EP=<n>` to watch it again. The viewer
runs until you close the window (`--episodes N` to cap it).

> **Display note:** the interactive viewer uses GLFW and needs `$DISPLAY`. If you
> have `MUJOCO_GL=egl` exported globally, unset it for `make watch` (EGL is
> offscreen-only). The `make video` path is the one that *wants* `MUJOCO_GL=egl`.

---

## What makes it sim2real (not just sim)

| Concern | How it's handled |
|---|---|
| **Action interface** | Policy outputs joint **position targets** (delta by default), exactly how STS3215 servos are driven in position mode. A sim action *is* a servo goal-position. |
| **Observation realizability** | Obs = joint angles + velocities (from the servos) + TCP (forward kinematics) + target. Nothing that can't be measured on the real arm. |
| **Dynamics gap** | Per-reset randomization of link mass/inertia, joint damping/friction/armature, actuator gains (kp/kv), gravity — see `DRConfig`. |
| **Command-path gap** | Randomized action latency + command noise model the Feetech serial bus (`sim2real/wrappers/`). |
| **Normalization gap** | VecNormalize stats are **baked into the exported ONNX**, so the robot feeds raw observations and can't desync. |
| **Kinematics gap** | The real robot computes TCP with the *same* MJCF used in training (`sim2real/utils/kinematics.py`). One source of truth. |

Evaluate with `--randomize` to get the number that actually predicts transfer.

---

## Tasks

Select the task with `env_id` in the config file.

### reach — `SO101Reach-v0` (`configs/reach.yaml`)
Move the gripper TCP to a random 3D target. Observation = joint angles +
velocities + TCP (forward kinematics) + target — **all measurable on the real
arm**, so this task transfers from proprioception alone.

### pick-and-place — `SO101PickPlace-v0` (`configs/pickplace.yaml`)
Grasp a free-standing cup **of fluid** and place it upright on a goal pad. The
gripper joins the action space (6 controlled joints); the cup is a rigid body
with a cosmetic half-fill (MuJoCo has no fluid sim, so *spilling is modelled as
a tilt limit*: tilting the cup past `spill_tilt` ≈ 26° ends the episode as a
failure, dense `w_upright` shaping keeps it level in between, and "placed"
requires the cup upright). Reward is staged with positive shaped bonuses:
approach → touch → lift → hold → carry → place, all while keeping the cup
level (design rationale and the reward-hacking war stories are in
[`docs/HOW_IT_WORKS.md` §5](docs/HOW_IT_WORKS.md)).

Two things make this task *trainable* at all:

- **Jaw collision fix** (`fix_gripper_collision`, default on): MuJoCo collides
  meshes as convex hulls, which seals the stock SO-101 gripper mouth solid —
  grasping is physically impossible against the vendored meshes. The env
  rebuilds the jaw collision as vertex-fitted box pads at load time
  (`docs/HOW_IT_WORKS.md` §2.5).
- **Curriculum resets** (`grasp_init_prob` / `hold_init_prob` /
  `place_init_prob`): a fraction of *training* episodes start with the jaws
  around the cup, already holding it, or holding it above the goal. Eval always
  starts from scratch, so reported success rates measure the real task.

> **Sim2real caveat:** the observation includes the cup pose, which the servos
> cannot measure. Real deployment needs object perception (overhead camera /
> AprilTag) feeding the cup pose into the same observation slot — everything else
> transfers as usual. Grasping is also much harder to learn than reach; expect
> the full 5M-step budget.

---

## Project structure

```
sim2real/
├── assets/so101/            # vendored SO-101 MJCF + STL meshes (upstream, pristine)
│   ├── so101_new_calib.xml  #   robot definition (do not edit; re-sync w/ scripts/fetch_assets.sh)
│   ├── reach_scene.xml      #   reach task scene: robot + floor + mocap target  (ours)
│   └── pickplace_scene.xml  #   pick-place scene: robot + cup + goal pad        (ours)
├── configs/                 # reach.yaml, sac_reach.yaml, reach_hard_dr.yaml, pickplace.yaml
├── sim2real/
│   ├── config.py            # typed dataclass config (one YAML == one run)
│   ├── envs/base.py         # SO101MujocoBase: shared control / DR / collisions
│   ├── envs/so101_reach.py  # reach task (proprioception-only, fully sim2real)
│   ├── envs/so101_pickplace.py # pick-and-place task (gripper + graspable cup)
│   ├── wrappers/            # command-path DR (latency, action noise)
│   ├── env_factory.py       # env/vecenv/normalization wiring
│   ├── train.py             # SB3 PPO/SAC training entrypoint
│   ├── eval.py              # success rate / distance metrics
│   ├── visualize.py         # interactive MuJoCo viewer / mp4 rendering
│   ├── export_policy.py     # -> ONNX with normalization baked in
│   ├── utils/kinematics.py  # shared forward kinematics (sim == real)
│   └── deploy/              # Feetech bus + real-robot control loop
├── tests/                   # fast CPU tests (env, DR, wrappers, FK, calibration)
├── scripts/                 # setup.sh, fetch_assets.sh
└── Makefile
```

## Configuration

Everything that changes a run lives in one YAML (`configs/reach.yaml`), grouped
into `env`, `dr`, `train`, `deploy`. Unknown keys fail loudly. See
`sim2real/config.py` for every field and its default. Common overrides are also
CLI flags: `--timesteps`, `--n-envs`, `--algo`, `--seed`, `--device`.

## Real-robot deployment

Deployment needs a one-time servo **calibration** (center ticks + direction per
joint) — see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the calibration,
safety limits, and wiring notes. Install the servo SDK with
`pip install -e ".[deploy]"`.

## Status & roadmap

- [x] SO-101 MuJoCo reach task + domain randomization
- [x] Pick-and-place task (gripper + graspable cup)
- [x] Gripper collision fix (convex-hull jaws) + grasp curricula — from-scratch
      grasping verified in sim
- [x] PPO/SAC training, eval, ONNX export, sim + hardware deploy loop
- [x] Visualization: interactive viewer, mp4 rendering, TensorBoard
- [ ] Consistent from-scratch place success (full pick→carry→place chain)
- [ ] Object perception for pick-place on hardware (camera / AprilTag)
- [ ] Pixel observations + camera domain randomization
- [ ] Sim2real gap logging (record real rollouts, compare to sim)

Model assets are © TheRobotStudio (SO-ARM100), Apache-2.0.
