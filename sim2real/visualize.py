"""Watch a trained SO-101 policy.

Two ways to see the policy after training:

    # interactive 3D viewer (needs a desktop session / display):
    python -m sim2real.visualize --run outputs/<run>
    python -m sim2real.visualize --run outputs/<run> --randomize   # under domain rand

    # headless: render an mp4 (works over SSH / no display):
    MUJOCO_GL=egl python -m sim2real.visualize --run outputs/<run> --video reach.mp4

The interactive viewer opens the MuJoCo window running the *same* MjData the
policy steps, and shows the red target sphere so you can watch the TCP chase it,
paced to the real control rate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from sim2real.config import Config, REPO_ROOT
from sim2real.env_factory import make_single_env


def _algo_cls(name: str):
    from stable_baselines3 import PPO, SAC
    return {"ppo": PPO, "sac": SAC}[name]


def _find_model(run_dir: Path, which: str) -> Path:
    for key in ([which] + ["best", "final"]):
        p = {"best": run_dir / "best_model" / "best_model.zip",
             "final": run_dir / "final_model.zip"}[key]
        if p.exists():
            return p
    raise FileNotFoundError(f"no model found under {run_dir}")


def _unwrap_base(venv):
    v = venv
    while hasattr(v, "venv"):
        v = v.venv
    return v.envs[0].unwrapped


def watch(run_dir: Path, episodes: int, randomize: bool, which: str,
          realtime: bool = True, seed: int | None = None, episode: int | None = None) -> None:
    """Open the interactive viewer and roll out the policy.

    Episode selection:
      * ``seed`` set        -> reproducible sequence; episode k uses seed ``seed+k``.
      * ``episode`` set     -> lock onto that one scenario and replay it (implies
                               a base seed of 0 if ``seed`` is None).
      * neither             -> fresh random episodes each run.
    """
    import mujoco.viewer
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = Config.from_yaml(run_dir / "config.yaml")
    venv = DummyVecEnv([make_single_env(cfg, randomize=randomize)])
    vn_path = run_dir / "vecnormalize.pkl"
    if vn_path.exists():
        venv = VecNormalize.load(str(vn_path), venv)
        venv.training = False
        venv.norm_reward = False

    model = _algo_cls(cfg.train.algo).load(str(_find_model(run_dir, which)), device="cpu")
    base = _unwrap_base(venv)
    dt = 1.0 / cfg.env.control_freq

    base_seed = 0 if (episode is not None and seed is None) else seed

    def ep_seed(k: int):
        if episode is not None:
            return base_seed + episode          # always the same scenario
        if base_seed is not None:
            return base_seed + k                # reproducible sequence
        return None                             # random

    def seeded_reset(k: int):
        s = ep_seed(k)
        if s is not None:
            venv.seed(s)                        # controls the next reset's scenario
        return venv.reset(), s

    label = (f"episode {episode} (looped)" if episode is not None
             else f"seeded from {base_seed}" if base_seed is not None else "random episodes")
    print(f"[watch] {cfg.train.algo} policy | {label} | randomize={randomize} | close the window to stop")

    obs, s = seeded_reset(0)
    ep, ep_ret = 0, 0.0
    with mujoco.viewer.launch_passive(base.model, base.data) as viewer:
        while viewer.is_running() and (episodes == 0 or ep < episodes):
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)
            ep_ret += float(reward[0])
            viewer.sync()
            if done[0]:
                idx = episode if episode is not None else ep
                seedtxt = f" seed={s}" if s is not None else ""
                print(f"  ep {idx:3d}{seedtxt}  success={infos[0].get('is_success')}  "
                      f"dist={infos[0].get('dist', 0)*1000:6.1f} mm  return={ep_ret:7.1f}")
                ep += 1
                ep_ret = 0.0
                obs, s = seeded_reset(ep)       # explicit seeded reset (overrides auto-reset)
            if realtime:
                sleep = dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    venv.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize an SO-101 reach policy")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--episodes", type=int, default=0,
                   help="stop after N episodes (0 = run until you close the window)")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; makes episodes reproducible (episode k uses seed+k)")
    p.add_argument("--episode", type=int, default=None,
                   help="watch only this episode index, replayed (implies --seed 0 if unset)")
    p.add_argument("--randomize", action="store_true", help="run under domain randomization")
    p.add_argument("--which", type=str, default="best", choices=["best", "final"])
    p.add_argument("--video", type=str, default=None,
                   help="headless: render an mp4 instead of opening a window (set MUJOCO_GL=egl)")
    p.add_argument("--no-realtime", action="store_true", help="run the viewer as fast as possible")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    if args.video:
        # reuse the tested eval recording path — no duplicated rollout logic
        from sim2real.eval import evaluate
        evaluate(run_dir, episodes=1, randomize=args.randomize, which=args.which, video=args.video)
    else:
        watch(run_dir, args.episodes, args.randomize, args.which,
              realtime=not args.no_realtime, seed=args.seed, episode=args.episode)
        # The MuJoCo/GLFW viewer's native teardown can segfault during normal
        # Python shutdown. We're done and everything is flushed, so exit hard to
        # skip the buggy finalizers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
