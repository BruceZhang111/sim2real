"""Live viewer for the training SAC policy on its learned curriculum skills.

Episodes cycle through place-starts / holding-starts / pregrasp-starts so the
window shows set-downs, carries and closes (from-scratch is still learning).
Close the window to stop.
"""
import time
import numpy as np
import gymnasium as gym
import mujoco.viewer
from dataclasses import replace
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sim2real  # noqa: F401
from sim2real.config import Config

import sys
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/sac_pickplace_1")
ck = sorted(RUN.glob("checkpoints/ckpt_*_steps.zip"),
            key=lambda p: int(p.stem.split("_")[1]))[-1]
n = ck.stem.split("_")[1]
cfg = Config.from_yaml(RUN / "config.yaml")
env_cfg = replace(cfg.env, place_init_prob=0.4, hold_init_prob=0.3,
                  grasp_init_prob=0.3, approach_init_prob=0.0)
dr = replace(cfg.dr, enabled=False, action_latency_steps=(0, 0), action_noise_std=0.0)

venv = DummyVecEnv([lambda: gym.make(env_cfg.env_id, config=env_cfg, dr=dr)])
venv = VecNormalize.load(str(RUN / f"checkpoints/ckpt_vecnormalize_{n}_steps.pkl"), venv)
venv.training = False
venv.norm_reward = False
model = SAC.load(str(ck), device="cpu")
base = venv.venv.envs[0].unwrapped
dt = 1.0 / cfg.env.control_freq

print(f"[show] SAC checkpoint {n} | starts: 40% at-goal, 30% holding, 30% jaws-around-cup")
print("[show] close the window to stop")
obs = venv.reset()
ep, ep_ret = 0, 0.0
with mujoco.viewer.launch_passive(base.model, base.data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = venv.step(action)
        ep_ret += float(reward[0])
        viewer.sync()
        if done[0]:
            i = infos[0]
            print(f"  ep {ep:3d}  success={i.get('is_success')}  spilled={i.get('spilled', False)}"
                  f"  tilt={np.degrees(i.get('tilt', 0)):4.0f} deg  return={ep_ret:7.1f}")
            ep += 1
            ep_ret = 0.0
        sleep = dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)
venv.close()
