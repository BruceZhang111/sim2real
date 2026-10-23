"""Run B1,B2,B3,C1,...,E3 serially, excluding completed group A.

python scripts/run_architecture_be.py --dry-run
python scripts/run_architecture_be.py
python scripts/run_architecture_be.py --start-at C2

Execution freezes all four configs before starting. Dry-run only reads configs.
Keep source code unchanged during the batch. No automatic resume or model reuse.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
from uuid import uuid4

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from sim2real.config import Config
from scripts.run_timestep_experiments import run_experiments

GROUPS = {
    "B": "mlp1440",
    "C": "simba64_512",
    "D": "simba256_512",
    "E": "simba64_256",
}
SEEDS = (1, 2, 3)
LABELS = tuple(f"{group}{seed}" for group in GROUPS for seed in SEEDS)


def load_configs() -> dict[str, Config]:
    """Reject a drifted recipe before spending time on any training."""
    configs = {}
    reference = None
    architecture_keys = {
        "policy_arch", "policy_hidden", "simba_actor_hidden_dim",
        "simba_actor_n_blocks", "simba_critic_hidden_dim", "simba_critic_n_blocks",
    }
    widths = {"C": (64, 512), "D": (256, 512), "E": (64, 256)}
    for group, name in GROUPS.items():
        path = REPOSITORY / "configs/architecture_8env" / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(path)
        cfg = Config.from_yaml(path)
        t = cfg.train
        if (t.algo != "ppo" or t.n_envs != 8 or t.target_kl != 0.33
                or t.total_timesteps != 2_000_000 or t.init_from is not None
                or cfg.env.reward_type != "revised_v2"):
            raise ValueError(f"{path}: expected fresh revised_v2 PPO, 8env/KL0.33/2M")
        if group == "B":
            if t.policy_arch != "mlp" or list(t.policy_hidden) != [1440, 1440]:
                raise ValueError(f"{path}: expected MLP[1440,1440]")
        elif (t.policy_arch != "simba"
              or (t.simba_actor_hidden_dim, t.simba_critic_hidden_dim) != widths[group]
              or (t.simba_actor_n_blocks, t.simba_critic_n_blocks) != (1, 2)
              or not t.normalize_obs):
            raise ValueError(f"{path}: unexpected SimBa architecture/normalization")
        if Path(t.log_dir).is_absolute() or Path(t.log_dir) != Path("outputs"):
            raise ValueError(f"{path}: expected log_dir=outputs")
        t.run_name = None
        common = asdict(cfg)
        for key in architecture_keys | {"seed"}:
            common["train"].pop(key)
        if reference is not None and common != reference:
            raise ValueError(f"{path}: non-architecture settings differ across groups")
        reference = common
        configs[group] = cfg
    return configs


def build_commands(paths: dict[str, Path], batch_id: str,
                   start_at: str = "B1") -> list[list[str]]:
    """Keep explicit seed and group+seed addname on every subprocess command."""
    commands = [
        [sys.executable, "-u", "-m", "sim2real.train",
         "--config", str(paths[group]), "--seed", str(seed),
         "--run-name", f"ppo_reach_{batch_id}", "--addname", f"{group}{seed}"]
        for group in GROUPS for seed in SEEDS
    ]
    return commands[LABELS.index(start_at):]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print source-config commands; no files or training")
    parser.add_argument("--start-at", choices=LABELS, default="B1",
                        help="first trial to launch; preceding trials are omitted, not resumed")
    args = parser.parse_args(argv)
    try:
        configs = load_configs()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    batch_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    sources = {g: REPOSITORY / "configs/architecture_8env" / f"{name}.yaml"
               for g, name in GROUPS.items()}
    if args.dry_run:
        print("[BE] Preview uses existing source YAML; execution freezes copies first.")
        for command in build_commands(sources, batch_id, args.start_at):
            print(shlex.join(command))
        return 0

    # Unique prefix protects existing A/B/... outputs, including restarted batches.
    batch_dir = REPOSITORY / "outputs" / f"architecture_be_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for group, cfg in configs.items():
        path = batch_dir / f"{group}_{GROUPS[group]}.yaml"
        cfg.to_yaml(path)
        snapshots[group] = path
    commands = build_commands(snapshots, batch_id, args.start_at)
    (batch_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(f"[BE] Batch plan/configs: {batch_dir}", flush=True)
    print(f"[BE] {len(commands)} sequential fresh trainings; A is excluded.", flush=True)
    print("[BE] Run suffixes are group+seed (B1...E3). Keep code unchanged.", flush=True)
    return run_experiments(commands)


if __name__ == "__main__":
    raise SystemExit(main())
