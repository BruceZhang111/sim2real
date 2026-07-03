# Deploying an SO-101 policy to the real arm

The deploy loop (`sim2real/deploy/deploy_so101.py`) is backend-agnostic: it runs
against a MuJoCo stand-in (`SimPlantBus`) or the real Feetech bus (`FeetechBus`)
through the same interface. Validate in sim first, then flip to hardware.

> Safety first: keep the workspace clear, keep a hand near the power switch, and
> always do a `--steps 20` dry run at a **low** `max_step_rad` before a full run.

---

## 1. Install the servo SDK

```bash
conda activate sim2real
pip install -e ".[deploy]"      # adds feetech-servo-sdk (scservo_sdk)
```

Give yourself serial access (once): `sudo usermod -aG dialout $USER` then log out/in.
Confirm the bus shows up (usually `/dev/ttyACM0`): `ls /dev/ttyACM* /dev/ttyUSB*`.

---

## 2. Calibrate the servos (one-time, critical)

The policy thinks in **radians about each joint's MJCF zero**. The servos think
in **ticks** (0–4095). Calibration is the mapping between them, expressed per
joint in `config.deploy`:

- `center_ticks[j]` — the tick reading when joint `j` is at **0 rad** (the MJCF
  home pose: arm pointing straight "forward/up" as in `so101_new_calib.xml`).
- `joint_sign[j]` — `+1` or `-1` so that **increasing radians** in sim matches
  **increasing angle in the physical joint's positive direction** (the axis in
  the MJCF). Get this wrong and the joint runs away from the target.
- `ticks_per_rev` — 4096 for STS3215 (leave as-is).

Conversion (`sim2real/deploy/feetech_bus.py`):

```
rad   = joint_sign * (tick - center_ticks) * 2π / ticks_per_rev
tick  = center_ticks + joint_sign * rad * ticks_per_rev / 2π
```

### How to get the numbers

1. Power the arm, torque **off**. Hand-move each joint to the MJCF zero pose and
   read its present tick with the `~/ServoControl` GUI (or `bus.read_ticks()` in
   a Python shell). That tick is `center_ticks[j]`.
2. From zero, rotate the joint a known **+** direction (matching the MJCF axis).
   If the tick value **increased**, `joint_sign[j] = +1`; if it **decreased**,
   `-1`.
3. Cross-check against sim: at qpos=0 the TCP is ≈ `[0.39, 0, 0.23] m`. If you
   command the home pose and the arm strikes that pose, your centers are right.

If you already calibrated the arm with LeRobot, reuse those homing offsets:
`center_ticks = homing_offset`, and set `joint_sign` from LeRobot's `drive_mode`.

Put the values in your run's `config.yaml` under `deploy:` (or a copy), e.g.:

```yaml
deploy:
  port: /dev/ttyACM0
  center_ticks: [2048, 2050, 2039, 2048, 2047, 2100]
  joint_sign:   [1, -1, 1, 1, -1, 1]
  max_step_rad: 0.10
  control_freq: 25.0
```

---

## 3. Dry-run in sim (no hardware)

```bash
python -m sim2real.deploy.deploy_so101 --run outputs/<run> --target 0.25 0.05 0.2
```

This drives `SimPlantBus` (a MuJoCo model) through the exact deploy code path:
observation build → FK → ONNX policy → safety clamps → tick conversion. A trained
policy should drive `dist` down toward the success threshold. If it does not, the
problem is the policy/obs, not the hardware.

---

## 4. Run on hardware

```bash
python -m sim2real.deploy.deploy_so101 --run outputs/<run> --port /dev/ttyACM0 \
    --target 0.25 0.05 0.2 --steps 100
```

Start-up sequence the script performs: open port → enable torque → loop at
`control_freq`, reading present positions, running the policy, writing goal
positions. On `Ctrl-C` or exit it **disables torque and closes the bus**.

### Safety layers (in code)

- `max_step_rad` clamps how far any joint can be commanded per control step.
- Joint targets are clipped to the MJCF joint limits.
- The gripper (if not actuated by the policy) is held at `env.gripper_hold`.
- Velocity is finite-differenced from position reads (no dependence on a noisy
  speed register).

Tune down `max_step_rad` (e.g. 0.05) for the first hardware runs, then raise it.

---

## 5. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Joint drives to a limit and stays | wrong `joint_sign[j]` |
| Arm reaches a mirrored/offset pose | wrong `center_ticks[j]` |
| Jerky / unstable motion | `max_step_rad` too high, or control rate ≠ training `control_freq` |
| Works in sim, bad on hardware | widen domain randomization (`configs/reach_hard_dr.yaml`) and retrain |
| `could not open port` | wrong `--port`, or missing `dialout` group |
| Import error `scservo_sdk` | `pip install -e ".[deploy]"` |

When sim is good but hardware is not, the fix is almost always **more domain
randomization + retrain**, not more code.
