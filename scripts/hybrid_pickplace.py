"""Hybrid pick-and-place: scripted approach+grasp, learned carry+place.

Pure-RL policies in this repo learn everything after first contact (grasp,
carry, place) but not the cold-start approach — the spill risk teaches
avoidance. On hardware the approach is trivial anyway: perception gives the
cup pose and the arm drives above it. This runner mirrors that architecture in
sim: a joint-space scripted primitive (IK pregrasp -> descend -> close ->
lift) followed by the SAC policy for transport and set-down.

Usage:
    python scripts/hybrid_pickplace.py [run_dir] [--video out.mp4] [--episodes N]
"""
import argparse
import time
from dataclasses import replace, fields
from pathlib import Path

import numpy as np
import gymnasium as gym
import mujoco

import sim2real  # noqa: F401
from sim2real.config import Config


def ik_fingers_down(model, data, env, target, iters=400):
    """DLS IK on the 5 arm joints: TCP site at target, fingers straight down."""
    site = env._tcp_site_id
    grip_bid = env._bid("gripper")
    arm_idx, arm_dof = env._qadr[:5], env._vadr[:5]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        e_pos = target - data.site_xpos[site]
        z_cur = -data.xmat[grip_bid].reshape(3, 3)[:, 2]
        e_rot = np.cross(z_cur, [0.0, 0.0, -1.0])
        if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 1e-3:
            break
        mujoco.mj_jacSite(model, data, jacp, None, site)
        mujoco.mj_jacBody(model, data, None, jacr, grip_bid)
        J = np.vstack([jacp[:, arm_dof], 0.35 * jacr[:, arm_dof]])
        e = np.concatenate([e_pos, 0.35 * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        data.qpos[arm_idx] = np.clip(data.qpos[arm_idx] + 0.4 * dq,
                                     env._jnt_range[:5, 0], env._jnt_range[:5, 1])
    return data.qpos[arm_idx].copy()


def scripted_action(env, q_target, g_cmd):
    """One env action stepping the arm joints toward q_target (P in joint space)."""
    q = env.data.qpos[env._qadr[:5]]
    a = np.zeros(env.n_action, dtype=np.float32)
    a[:5] = np.clip((q_target - q) / env.action_scales[:5], -1.0, 1.0)
    a[5] = g_cmd
    return a


def run_episode(venv, env, model_rl, record=None, viewer=None, dt=0.0,
                max_policy_steps=280):
    """Scripted pickup then policy hand-off. Returns (success, spilled, frames)."""
    obs = venv.reset()
    frames = []

    def snap():
        if viewer is not None:
            viewer.sync()
            time.sleep(dt)
        if record is not None:
            cam, renderer = record
            cam.lookat[:] = [0.24, 0.0, 0.05]
            cam.distance, cam.azimuth, cam.elevation = 0.55, 150, -25
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())

    # -- plan the grasp pose on a scratch MjData (leaves the episode untouched)
    cup = env._cup()
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    # two IK passes: the TCP site sits on the FIXED JAW TIP, but the cup must
    # end up mid-mouth, ~30 mm from the site along the gripper's x axis —
    # first pass gets the orientation, second aims the offset target
    ik_fingers_down(env.model, scratch, env, np.array([cup[0], cup[1], 0.014]))
    mujoco.mj_forward(env.model, scratch)
    rot = scratch.xmat[env._bid("gripper")].reshape(3, 3)
    target = np.array([cup[0], cup[1], 0.014]) - 0.030 * rot[:, 0]
    target[2] = 0.014
    q_grasp = ik_fingers_down(env.model, scratch, env, target)
    q_hover = q_grasp.copy()
    q_hover[1] -= 0.35
    q_hover[3] += 0.35   # wrist keeps fingers down while raised
    q_lift = q_hover

    def drive(q_target, g_cmd, steps):
        nonlocal obs
        done_info = None
        for _ in range(steps):
            a = scripted_action(env, q_target, g_cmd)
            obs, _, dones, infos = venv.step(a[None, :])
            snap()
            if dones[0]:
                done_info = infos[0]
                break
        return done_info

    # phase A: scripted — hover above cup, descend, close, lift
    for q_t, g, n in ((q_hover, 1.0, 45), (q_grasp, 1.0, 40),
                      (q_grasp, -1.0, 10), (q_lift, -1.0, 25)):
        info = drive(q_t, g, n)
        if info is not None:                      # spill/loss during pickup
            return False, bool(info.get("spilled")), frames
    if not env._get_info()["holding"]:
        return False, False, frames               # pickup failed, no hand-off

    # phase B: learned — carry and place
    for _ in range(max_policy_steps):
        act, _ = model_rl.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(act)
        snap()
        i = infos[0]
        if i.get("is_success"):
            for _ in range(30):                   # dwell on the placed cup
                act, _ = model_rl.predict(obs, deterministic=True)
                obs, _, dones, infos = venv.step(act)
                snap()
                if dones[0]:
                    break
            return True, False, frames
        if dones[0]:
            return False, bool(i.get("spilled")), frames
    return False, False, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="outputs/sac_pickplace_5")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--watch", action="store_true",
                   help="open the live MuJoCo window (needs a display)")
    args = p.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run = Path(args.run)
    cfg = Config.from_yaml(run / "config.yaml")
    zeros = {f.name: 0.0 for f in fields(cfg.env) if f.name.endswith("_init_prob")}
    env_cfg = replace(cfg.env, **zeros)           # the true task, no curriculum
    dr = replace(cfg.dr, enabled=False, action_latency_steps=(0, 0), action_noise_std=0.0)
    venv = DummyVecEnv([lambda: gym.make(env_cfg.env_id, config=env_cfg, dr=dr)])
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False
    env = venv.venv.envs[0].unwrapped
    model_rl = SAC.load(str(run / "final_model.zip"), device="cpu")

    record = None
    kept = []
    if args.video:
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        record = (mujoco.MjvCamera(), renderer)

    viewer_cm = None
    viewer = None
    dt = 0.0
    if args.watch:
        import mujoco.viewer
        viewer_cm = mujoco.viewer.launch_passive(env.model, env.data)
        viewer = viewer_cm.__enter__()
        dt = 1.0 / cfg.env.control_freq

    wins = spills = 0
    for ep in range(args.episodes):
        if viewer is not None and not viewer.is_running():
            break
        venv.seed(9000 + ep)
        t0 = time.time()
        ok, spilled, frames = run_episode(venv, env, model_rl, record=record,
                                          viewer=viewer, dt=dt)
        wins += ok
        spills += spilled
        print(f"ep {ep:2d}: {'SUCCESS' if ok else 'fail'}"
              f"{' (spilled)' if spilled else ''}  [{time.time()-t0:.1f}s]")
        if ok and args.video and len(kept) < 3:
            kept.extend(frames)
    if viewer_cm is not None:
        viewer_cm.__exit__(None, None, None)
    print(f"\nhybrid task success: {wins}/{args.episodes} (spills {spills})")

    if args.video and kept:
        import imageio
        imageio.mimwrite(args.video, kept, fps=int(cfg.env.control_freq), quality=8)
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
