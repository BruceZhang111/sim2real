"""Train a sim2real reach policy for the SO-101 (Stable-Baselines3).

Usage
-----
    python -m sim2real.train --config configs/reach.yaml
    python -m sim2real.train --config configs/reach.yaml --timesteps 50000 --n-envs 4
    python -m sim2real.train --smoke      # tiny run to verify the pipeline

A run directory ``outputs/<run_name>/`` is created containing tensorboard logs,
periodic checkpoints, the best model, the final model, the VecNormalize stats
(needed at deploy time) and a snapshot of the exact config used.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

from sim2real.config import Config, REPO_ROOT


def build_model(cfg: Config, venv, tb_dir: str):
    from stable_baselines3 import PPO, SAC

    policy_kwargs = dict(net_arch=list(cfg.train.policy_hidden))
    if cfg.train.algo == "ppo":
        policy_kwargs["log_std_init"] = cfg.train.log_std_init
    common = dict(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        seed=cfg.train.seed,
        device=cfg.train.device,
        gamma=cfg.train.gamma,
        learning_rate=cfg.train.learning_rate,
        tensorboard_log=tb_dir,
        policy_kwargs=policy_kwargs,
    )
    if cfg.train.algo == "ppo":
        return PPO(
            **common,
            n_steps=cfg.train.n_steps,
            batch_size=cfg.train.batch_size,
            n_epochs=cfg.train.n_epochs,
            gae_lambda=cfg.train.gae_lambda,
            clip_range=cfg.train.clip_range,
            ent_coef=cfg.train.ent_coef,
        )
    if cfg.train.algo == "sac":
        return SAC(
            **common,
            buffer_size=cfg.train.buffer_size,
            learning_starts=cfg.train.learning_starts,
            batch_size=cfg.train.batch_size,
            train_freq=cfg.train.train_freq,
            tau=cfg.train.tau,
        )
    raise ValueError(f"unknown algo {cfg.train.algo!r} (use 'ppo' or 'sac')")


class NormalizedEvalCallback:
    """Factory for an EvalCallback that syncs obs normalization before eval.

    Eval envs use their own VecNormalize with ``training=False``; without
    copying the trainer's running mean/var the eval observations would be
    normalized with stale stats and the reported success rate would be wrong.
    """

    @staticmethod
    def make(cfg: Config, eval_venv, best_dir: str, log_dir: str):
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import VecNormalize

        eval_freq = max(cfg.train.eval_freq // cfg.train.n_envs, 1)

        class _Cb(EvalCallback):
            def _on_step(self) -> bool:
                if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                    train_env = self.model.get_vec_normalize_env()
                    if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                        self.eval_env.obs_rms = deepcopy(train_env.obs_rms)
                return super()._on_step()

        return _Cb(
            eval_venv,
            best_model_save_path=best_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=cfg.train.n_eval_episodes,
            deterministic=True,
            render=False,
        )


def train(cfg: Config) -> Path:
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from sim2real.env_factory import make_training_venv, make_eval_venv

    # task tag derived from the env id, e.g. SO101PickPlace-v0 -> "pickplace"
    task = cfg.env.env_id.replace("SO101", "").split("-")[0].lower() or "task"
    run_name = cfg.train.run_name or f"{cfg.train.algo}_{task}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = (REPO_ROOT / cfg.train.log_dir / run_name).resolve()
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    print(f"[train] run dir: {run_dir}")

    venv = make_training_venv(cfg)
    if cfg.train.init_from:
        # warm start: restore normalization stats first so the loaded policy
        # sees observations on the scale it was trained with
        from stable_baselines3.common.vec_env import VecNormalize

        ckpt = Path(cfg.train.init_from)
        vn_pkl = ckpt.with_name(ckpt.stem.replace("ckpt_", "ckpt_vecnormalize_") + ".pkl")
        if vn_pkl.exists() and isinstance(venv, VecNormalize):
            venv = VecNormalize.load(str(vn_pkl), venv.venv)
            print(f"[train] warm start: normalization from {vn_pkl.name}")
    eval_venv = make_eval_venv(cfg)

    model = build_model(cfg, venv, str(run_dir / "tensorboard"))
    if cfg.train.init_from:
        model.set_parameters(str(Path(cfg.train.init_from)))
        print(f"[train] warm start: policy weights from {cfg.train.init_from}")
        if hasattr(model.policy, "log_std"):
            # re-inflate exploration: a converged checkpoint carries a
            # collapsed log_std, and continuing from std~0 can't discover
            # anything the source run hadn't already learned
            import torch
            with torch.no_grad():
                model.policy.log_std.fill_(cfg.train.log_std_init)
            print(f"[train] warm start: log_std reset to {cfg.train.log_std_init}")

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(cfg.train.save_freq // cfg.train.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ckpt",
            save_vecnormalize=True,
        ),
        NormalizedEvalCallback.make(
            cfg, eval_venv, str(run_dir / "best_model"), str(run_dir / "eval")
        ),
    ])

    try:
        model.learn(total_timesteps=cfg.train.total_timesteps, callback=callbacks, progress_bar=True)
    finally:
        model.save(run_dir / "final_model")
        # VecNormalize stats are required to reproduce observations at deploy time
        vn = model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(run_dir / "vecnormalize.pkl"))
        venv.close()
        eval_venv.close()

    print(f"[train] done. artifacts in {run_dir}")
    return run_dir


def apply_overrides(cfg: Config, args) -> Config:
    if args.timesteps is not None:
        cfg.train.total_timesteps = args.timesteps
    if args.n_envs is not None:
        cfg.train.n_envs = args.n_envs
    if args.algo is not None:
        cfg.train.algo = args.algo
    if args.run_name is not None:
        cfg.train.run_name = args.run_name
    if args.init_from is not None:
        cfg.train.init_from = args.init_from
    if args.device is not None:
        cfg.train.device = args.device
    if args.seed is not None:
        cfg.train.seed = args.seed
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Train SO-101 sim2real reach policy")
    p.add_argument("--config", type=str, default="configs/reach.yaml")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--algo", type=str, default=None, choices=["ppo", "sac"])
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start from this checkpoint .zip (see TrainConfig.init_from)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = Config.from_yaml(config_path) if config_path.exists() else Config()

    if args.smoke:
        cfg.train.total_timesteps = 4000
        cfg.train.n_envs = 2
        cfg.train.eval_freq = 2000
        cfg.train.n_eval_episodes = 3
        cfg.train.save_freq = 4000
        cfg.train.run_name = "smoke"

    cfg = apply_overrides(cfg, args)
    train(cfg)


if __name__ == "__main__":
    main()
