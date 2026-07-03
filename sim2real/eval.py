"""Evaluate a trained SO-101 reach policy.

    python -m sim2real.eval --run outputs/ppo_reach_XXXX --episodes 50
    python -m sim2real.eval --run outputs/ppo_reach_XXXX --randomize   # test DR robustness
    MUJOCO_GL=egl python -m sim2real.eval --run outputs/... --video out.mp4

Reports success rate and TCP-to-target distance. ``--randomize`` evaluates
under domain randomization, which is the number that actually predicts
real-world transfer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim2real.config import Config, REPO_ROOT
from sim2real.env_factory import make_single_env


def _algo_cls(name: str):
    from stable_baselines3 import PPO, SAC
    return {"ppo": PPO, "sac": SAC}[name]


def _find_model(run_dir: Path, which: str) -> Path:
    candidates = {
        "best": run_dir / "best_model" / "best_model.zip",
        "final": run_dir / "final_model.zip",
    }
    path = candidates[which]
    if not path.exists():
        # fall back to whichever exists
        for p in candidates.values():
            if p.exists():
                return p
        raise FileNotFoundError(f"no model found under {run_dir}")
    return path


def _unwrap_base(venv):
    """Reach the underlying SO101ReachEnv through the vec/normalize stack."""
    v = venv
    while hasattr(v, "venv"):
        v = v.venv
    return v.envs[0].unwrapped


def evaluate(run_dir: Path, episodes: int, randomize: bool, which: str, video: str | None):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = Config.from_yaml(run_dir / "config.yaml")
    model_path = _find_model(run_dir, which)
    render_mode = "rgb_array" if video else None

    venv = DummyVecEnv([make_single_env(cfg, randomize=randomize, render_mode=render_mode)])
    vn_path = run_dir / "vecnormalize.pkl"
    if vn_path.exists():
        venv = VecNormalize.load(str(vn_path), venv)
        venv.training = False
        venv.norm_reward = False

    model = _algo_cls(cfg.train.algo).load(str(model_path), device="cpu")
    print(f"[eval] {model_path.name}  | randomize={randomize} | episodes={episodes}")

    successes, dists, returns = [], [], []
    frames = []
    base = _unwrap_base(venv) if video else None

    for ep in range(episodes):
        obs = venv.reset()
        done = np.array([False])
        ep_ret, last_info = 0.0, {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)
            ep_ret += float(reward[0])
            last_info = infos[0]
            if video and ep == 0:
                try:
                    frames.append(base.render())
                except Exception as e:  # headless without EGL, etc.
                    print(f"[eval] render unavailable ({e}); skipping video")
                    video = None
        successes.append(bool(last_info.get("is_success", False)))
        dists.append(float(last_info.get("dist", np.nan)))
        returns.append(ep_ret)

    print(f"[eval] success rate : {np.mean(successes)*100:.1f}%  ({sum(successes)}/{episodes})")
    print(f"[eval] final dist   : {np.nanmean(dists)*1000:.1f} mm  (median {np.nanmedian(dists)*1000:.1f})")
    print(f"[eval] mean return  : {np.mean(returns):.2f}")

    if video and frames:
        import imageio
        out = Path(video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=int(cfg.env.control_freq))
        print(f"[eval] saved video ({len(frames)} frames) -> {out}")

    venv.close()
    return dict(success_rate=float(np.mean(successes)), mean_dist=float(np.nanmean(dists)))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an SO-101 reach policy")
    p.add_argument("--run", type=str, required=True, help="run directory under outputs/")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--randomize", action="store_true", help="evaluate under domain randomization")
    p.add_argument("--which", type=str, default="best", choices=["best", "final"])
    p.add_argument("--video", type=str, default=None, help="path to save an mp4 of episode 0")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    evaluate(run_dir, args.episodes, args.randomize, args.which, args.video)


if __name__ == "__main__":
    main()
