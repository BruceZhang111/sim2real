"""Run independent 2M/3M/5M PPO experiments with one frozen configuration.

Run inside the WSL sim2real environment:
    python scripts/run_timestep_experiments.py --dry-run
    python scripts/run_timestep_experiments.py

Only the requested budget and output name differ between runs. This is not
checkpoint continuation. No model, simulator or torch import is needed here.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Support direct execution even when the package has not been installed editable.
REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from sim2real.config import Config, REPO_ROOT


TIMESTEP_BUDGETS = (2_000_000, 3_000_000, 5_000_000)


def load_config(path: Path, policy_arch: str) -> Config:
    """Load once so editing the source YAML cannot change later experiments."""
    cfg = Config.from_yaml(path)
    if cfg.train.algo != "ppo":
        raise ValueError("This budget experiment requires train.algo=ppo")
    if cfg.train.init_from is not None:
        raise ValueError("Set train.init_from=null: all three runs must start from scratch")
    cfg.train.policy_arch = policy_arch
    if policy_arch == "simba" and not cfg.train.normalize_obs:
        raise ValueError("SimBa requires train.normalize_obs=true")
    # Input run names must not merge independent trials into an existing run.
    cfg.train.run_name = None
    return cfg


def build_commands(cfg: Config, snapshot: Path) -> list[list[str]]:
    """Use one interpreter and one snapshot for all three training processes."""
    task = cfg.env.env_id.replace("SO101", "").split("-")[0].lower() or "task"
    return [
        [
            sys.executable, "-u", "-m", "sim2real.train",
            "--config", str(snapshot),
            "--timesteps", str(budget),
            "--run-name", f"ppo_{task}-{cfg.train.policy_arch}-ts{budget}",
        ]
        for budget in TIMESTEP_BUDGETS
    ]


def run_experiments(commands: list[list[str]]) -> int:
    """Run serially; a failed or interrupted trial must not start the next one."""
    for index, command in enumerate(commands, start=1):
        print(f"\n[sweep] Starting {index}/{len(commands)}: {shlex.join(command)}", flush=True)
        # Inherit terminal streams and process group: preserve progress display
        # and let terminal Ctrl+C reach the training process for its finally block.
        process = subprocess.Popen(command, cwd=REPO_ROOT)
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            print("\n[sweep] Interrupted; waiting for this run to exit. No further runs.", flush=True)
            process.wait()
            return 130
        if returncode != 0:
            print(f"[sweep] Run failed (exit {returncode}); remaining runs cancelled.", flush=True)
            return returncode if returncode > 0 else 128 - returncode
        print(f"[sweep] Completed {index}/{len(commands)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reach_forSimBa.yaml"))
    parser.add_argument("--policy-arch", choices=("mlp", "simba"), default="simba",
                        help="architecture for ALL runs (default: simba, overrides YAML)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without creating files or starting training")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        cfg = load_config(config_path, args.policy_arch)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    output_root = (REPO_ROOT / cfg.train.log_dir).resolve()
    batch_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    batch_dir = output_root / f"timestep_sweep_{batch_id}"
    cfg.train.log_dir = str(batch_dir)
    snapshot = batch_dir / "shared_config.yaml"
    commands = build_commands(cfg, snapshot)

    print(f"[sweep] Source: {config_path.resolve()}", flush=True)
    print(f"[sweep] Python: {sys.executable}", flush=True)
    print(f"[sweep] Output: {batch_dir}", flush=True)
    print(f"[sweep] arch={cfg.train.policy_arch}, seed={cfg.train.seed}, "
          f"n_envs={cfg.train.n_envs}, n_steps={cfg.train.n_steps}, "
          f"epochs={cfg.train.n_epochs}, lr={cfg.train.learning_rate}, "
          f"target_kl={cfg.train.target_kl}, DR={cfg.dr.enabled}", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
    print("[sweep] Independent fresh starts; only budget/name vary. "
          "PPO may exceed each budget to finish its rollout batch.", flush=True)
    if args.dry_run:
        return 0

    # Never reuse a previous batch directory or touch its training artifacts.
    batch_dir.mkdir(parents=True, exist_ok=False)
    cfg.to_yaml(snapshot)
    (batch_dir / "commands.txt").write_text(
        "\n".join(shlex.join(command) for command in commands) + "\n", encoding="utf-8"
    )
    print("[sweep] Configuration frozen. Keep source code unchanged during the sweep.", flush=True)
    returncode = run_experiments(commands)
    if returncode == 0:
        print(f"[sweep] All three experiments completed: {batch_dir}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
