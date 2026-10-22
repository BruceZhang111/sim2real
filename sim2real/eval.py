"""Deterministic, explicitly seeded evaluation with raw episode exports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sim2real.config import Config, REPO_ROOT
from sim2real.env_factory import make_single_env


def _algo_cls(name: str):
    from stable_baselines3 import PPO, SAC
    return {"ppo": PPO, "sac": SAC}[name]


def _find_model(run_dir: Path, which: str) -> Path:
    path = {"best": run_dir / "best_model/best_model.zip",
            "final": run_dir / "final_model.zip"}[which]
    if not path.is_file():
        raise FileNotFoundError(f"requested {which} model missing: {path}; no fallback")
    return path


def file_hash(path: Path) -> str:
    """Hash artifacts so exports identify the actual bytes evaluated."""
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _normalization_path(run_dir: Path, which: str, cfg: Config) -> Path | None:
    if not (cfg.train.normalize_obs or cfg.train.normalize_reward):
        return None
    path = run_dir / ("best_model/vecnormalize.pkl" if which == "best" else "vecnormalize.pkl")
    if not path.is_file():
        raise FileNotFoundError(f"paired normalization missing: {path}; historical best cannot use final stats")
    if which == "best" and not (run_dir / "best_model/best_pair.json").is_file():
        raise FileNotFoundError("best pair manifest missing; cannot verify historical best")
    return path


def _unwrap_base(venv):
    while hasattr(venv, "venv"):
        venv = venv.venv
    return venv.envs[0].unwrapped


def _provenance() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    return {
        "code_commit": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        "code_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "torch", "mujoco", "gymnasium", "stable-baselines3")},
    }


def evaluate(run_dir: Path, episodes: int, randomize: bool, which: str = "final",
             video: str | None = None, eval_seed: int = 20000,
             output: Path | None = None) -> dict:
    """Reset each episode with eval_seed+i; never update saved normalization.

    Seeds are shared across models within each mode. Nominal and DR consume
    different RNG draws and are separate test strata, not paired target scenes.
    Smoothness measures policy commands before latency/noise, with a_(-1)=0.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if episodes < 1 or eval_seed < 0 or eval_seed + episodes - 1 > 2**32 - 1:
        raise ValueError("episodes must be positive and seeds must fit uint32")
    run_dir = run_dir.resolve()
    cfg = Config.from_yaml(run_dir / "config.yaml")
    model_path = _find_model(run_dir, which)
    vn_path = _normalization_path(run_dir, which, cfg)
    mode = "dr" if randomize else "nominal"
    output = Path(output) if output else run_dir / "eval_raw" / (
        f"{which}_{mode}_seed{eval_seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
    # Reserve an immutable export directory before any simulation work.
    metadata = dict(schema_version=1, which=which, randomize=randomize,
                    episodes=episodes, eval_seed=eval_seed, deterministic=True,
                    seed_protocol="reset(seed=eval_seed+episode); nominal/DR separate RNG strata",
                    smoothness="mean(sum((command_t-command_prev)^2)); prev starts at zero",
                    model_path=str(model_path), model_sha256=file_hash(model_path),
                    normalization_path=str(vn_path) if vn_path else None,
                    normalization_sha256=file_hash(vn_path) if vn_path else None,
                    pairing="run-relative legacy final" if which == "final" else "best sibling",
                    config_sha256=file_hash(run_dir / "config.yaml"),
                    created_utc=datetime.now(timezone.utc).isoformat(), **_provenance())
    manifest = model_path.parent / ("best_pair.json" if which == "best" else "final_pair.json")
    if vn_path and manifest.exists():
        pair = json.loads(manifest.read_text())
        if pair["model_sha256"] != metadata["model_sha256"] or pair["normalization_sha256"] != metadata["normalization_sha256"]:
            raise ValueError(f"{which} pair hash mismatch")
        metadata["pairing"] = f"{which} hash manifest verified"
    output.mkdir(parents=True, exist_ok=False)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    venv = DummyVecEnv([make_single_env(cfg, randomize=randomize,
                                      render_mode="rgb_array" if video else None)])
    rows, frames = [], []
    try:
        if vn_path:
            venv = VecNormalize.load(str(vn_path), venv)
            venv.training = False
            venv.norm_reward = False
        model = _algo_cls(cfg.train.algo).load(str(model_path), device="cpu")
        fields = ["episode", "seed", "success", "final_dist_m", "return",
                  "length_steps", "duration_s", "command_delta_sq_mean",
                  "command_sq_mean", "terminated", "truncated"]
        with (output / "episodes.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for ep in range(episodes):
                venv.seed(eval_seed + ep)
                obs = venv.reset()
                initial_info = _unwrap_base(venv)._get_info()
                previous = np.zeros(venv.action_space.shape)
                ep_return = delta_sq = command_sq = 0.0
                length = 0
                done = [False]
                while not done[0]:
                    action, _ = model.predict(obs, deterministic=True)
                    delta_sq += float(np.sum((action[0] - previous)**2))
                    command_sq += float(np.sum(action[0]**2))
                    previous = action[0].copy()
                    obs, reward, done, infos = venv.step(action)
                    ep_return += float(reward[0])
                    length += 1
                    if video and ep == 0 and not done[0]:
                        frames.append(_unwrap_base(venv).render())
                info = infos[0]  # terminal info, not the automatically reset state
                truncated = bool(info.get("TimeLimit.truncated", False))
                row = dict(episode=ep, seed=eval_seed + ep,
                           success=bool(info.get("is_success", False)),
                           final_dist_m=float(info["dist"]), return_=ep_return,
                           length_steps=length, duration_s=length/cfg.env.control_freq,
                           command_delta_sq_mean=delta_sq/length, command_sq_mean=command_sq/length,
                           terminated=not truncated, truncated=truncated)
                row["return"] = row.pop("return_")
                writer.writerow(row)
                stream.flush()
                # Include the sampled target for auditing equal scenes across models.
                rows.append({**row, "initial_target_m": np.asarray(initial_info["target"]).tolist()})
        summary = dict(
            success_rate=float(np.mean([r["success"] for r in rows])),
            mean_dist=float(np.mean([r["final_dist_m"] for r in rows])),
            median_dist=float(np.median([r["final_dist_m"] for r in rows])),
            mean_return=float(np.mean([r["return"] for r in rows])),
            mean_length=float(np.mean([r["length_steps"] for r in rows])),
            mean_command_delta_sq=float(np.mean([r["command_delta_sq_mean"] for r in rows])))
        (output / "results.json").write_text(json.dumps(dict(metadata=metadata, summary=summary,
                                                            episodes=rows), indent=2) + "\n")
        print(f"[eval] {which} {mode} seed={eval_seed} episodes={episodes}")
        print(f"[eval] {json.dumps(summary)}")
        print(f"[eval] raw export: {output}")
        if video and frames:
            import imageio
            out = Path(video)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(out, frames, fps=int(cfg.env.control_freq))
        return summary
    finally:
        venv.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--which", default="final", choices=["best", "final"])
    p.add_argument("--eval-seed", type=int, default=20000)
    p.add_argument("--output", type=Path, help="new export directory; existing paths rejected")
    p.add_argument("--video")
    args = p.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    evaluate(run_dir, args.episodes, args.randomize, args.which, args.video,
             args.eval_seed, args.output)


if __name__ == "__main__":
    main()
