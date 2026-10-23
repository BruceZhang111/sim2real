"""固定随机种子评估 SO-101 模型，并导出逐 episode 原始结果。

终端用法（在服务器执行；只评估，不训练）
----------------------------------------
先进入仓库并激活环境，把 RUN 改成实际训练目录：
    cd /home/zyg/Projects/sim2real
    source .venv/bin/activate
    RUN='outputs/你的训练目录名'

1. final 模型，nominal 名义动力学，200 回合：
    python -m sim2real.eval --run "$RUN" --which final --episodes 200 --eval-seed 20000

2. 同一 final 模型，DR 鲁棒性，200 回合：
    python -m sim2real.eval --run "$RUN" --which final --episodes 200 --eval-seed 20000 --randomize

3. 快速检查（仅 2 回合；不能作为正式性能结果）：
    python -m sim2real.eval --run "$RUN" --which final --episodes 2 --eval-seed 20000

4. 显式指定新的结果目录（该目录必须不存在）：
    python -m sim2real.eval --run "$RUN" --which final --episodes 200 --eval-seed 20000 --output "$RUN/eval_raw/manual_nominal_seed20000"

5. 评估 best（仅限具备同一时点归一化文件和配对清单的模型）：
    python -m sim2real.eval --run "$RUN" --which best --episodes 200 --eval-seed 20000
    python -m sim2real.eval --run "$RUN" --which best --episodes 200 --eval-seed 20000 --randomize

6. 查看全部参数：
    python -m sim2real.eval --help

本地 WSL 也使用相同评估命令；先激活已有 sim2real Python 环境并切换仓库：
    cd '/home/brucezhang/Dev/Projects/SO101 Sim2Real/sim2real'

参数与结果说明
--------------
* --run 指向训练目录，不是 final_model.zip。目录需要 config.yaml，以及所选模型
  和配套归一化文件。相对 --run 从仓库根目录解析；--output 相对当前终端目录解析。
* 默认 --which final、--episodes 200、--eval-seed 20000；不加 --randomize 就是
  nominal。--randomize 使用 run 的 DR 配置；若该配置关闭 DR，不会强制开启。
* 第 i 回合使用 20000+i（i 从 0 开始）；重新运行仍从 20000 开始。训练 seed1/2/3
  与此独立，各模型主比较使用相同评估 seed。nominal/DR 为独立分层，不是逐目标配对。
* 推理为 deterministic=True、CPU，冻结观测归一化，reward 不归一化。
* final 读取 run 根目录的 final_model.zip / vecnormalize.pkl；best 读取
  best_model/best_model.zip、best_model/vecnormalize.pkl、best_model/best_pair.json。
  历史 best 缺少配对文件会报错，不回退到 final 或借用 final 统计量。
* 默认自动创建 RUN/eval_raw 下带模型、模式、seed、时间戳的新目录。两种模式会分开
  保存；终端打印 raw export 实际路径。每次独立评估，不覆盖旧结果目录。
* metadata.json：seed、模式、文件哈希、代码/依赖版本。
  episodes.csv：逐回合 success、final_dist_m、return、length_steps、duration_s、
  command_delta_sq_mean、command_sq_mean、terminated/truncated。
  results.json：汇总、逐回合记录和初始目标。
* 汇总 success_rate 为 0~1 比例（乘 100 为百分比），mean_dist / median_dist 单位
  为米（乘 1000 为毫米），mean_length 为控制步数。命令差分指标不是实际关节速度。
* 可选 --video /路径/episode0.mp4 保存首回合视频；无显示服务器需可用的渲染后端。
  正式 nominal/DR 数值评估无需开启视频或图形界面。
"""
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
