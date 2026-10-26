#!/usr/bin/env python3
"""Export raw eval/rollout success-rate scalars without running environments.

From the repository root on the server:
    python scripts/export_success_rates.py --output-dir exports/success_rates_20261026

The default 21 run names match the user's screenshot (seeds 0, 1, 3).
Override them with --runs RUN_NAME [RUN_NAME ...].
Inputs are read-only. The destination must not already exist. No smoothing,
interpolation, seed averaging, training, or model evaluation is performed.

For overlapping TensorBoard files, identical (tag, step) values are collapsed.
Conflicting values at the same step are rejected: select the desired logging
session rather than silently combining resumed/restarted histories.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TAGS = ("eval/success_rate", "rollout/success_rate")
DEFAULT_RUNS = (
    "ppo_reach_20261026_135115-MLP256seed0",
    "ppo_reach_20261026_135134-MLP256seed1",
    "ppo_reach_20261026_135154-MLP256seed3",
    "ppo_reach_20261026_135229-MLP1440seed0",
    "ppo_reach_20261026_135246-MLP1440seed1",
    "ppo_reach_20261026_135304-MLP1440seed3",
    "ppo_reach_20261026_135428-actor64-critic512-seed0",
    "ppo_reach_20261026_135440-actor64-critic512-seed1",
    "ppo_reach_20261026_135449-actor64-critic512-seed3",
    "ppo_reach_20261026_145401-actor64-critic256-seed0",
    "ppo_reach_20261026_145409-actor64-critic256-seed1",
    "ppo_reach_20261026_145418-actor64-critic256-seed3",
    "ppo_reach_20261026_145450-actor128-critic512-seed0",
    "ppo_reach_20261026_145504-actor128-critic512-seed1",
    "ppo_reach_20261026_145516-actor128-critic512-seed3",
    "ppo_reach_20261026_171051-MLP1024-seed0",
    "ppo_reach_20261026_171143-MLP1024-seed1",
    "ppo_reach_20261026_171153-MLP1024-seed3",
    "ppo_reach_20261026_171210-MLP512-seed0",
    "ppo_reach_20261026_171223-MLP512-seed1",
    "ppo_reach_20261026_171235-MLP512-seed3",
)
FIELDS = ("run", "metric", "timesteps", "success_rate", "wall_time", "source_file")


def read_run(run_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read every scalar from each event file, retaining independent sessions."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted((run_dir / "tensorboard").rglob("events.out.tfevents.*"))
    if not files:
        raise ValueError(f"No TensorBoard event files: {run_dir / 'tensorboard'}")
    points: dict[str, dict[int, dict[str, Any]]] = {tag: {} for tag in TAGS}
    duplicate_count = 0
    for event_file in files:
        # Unlimited scalar retention avoids EventAccumulator's sampling default.
        # Read files separately so restart purging cannot silently discard history.
        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0},
            purge_orphaned_data=False,
        )
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for tag in TAGS:
            if tag not in available:
                continue
            for event in accumulator.Scalars(tag):
                value = float(event.value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Invalid success rate {value}: {event_file}, {tag}, step {event.step}"
                    )
                row = dict(
                    run=run_dir.name, metric=tag, timesteps=int(event.step),
                    success_rate=value, wall_time=float(event.wall_time),
                    source_file=str(event_file.relative_to(run_dir)),
                )
                old = points[tag].get(row["timesteps"])
                if old is not None:
                    if old["success_rate"] != value:
                        raise ValueError(
                            f"Conflicting {tag} at step {event.step} in {run_dir.name}: "
                            f"{old['success_rate']} ({old['source_file']}) vs "
                            f"{value} ({row['source_file']}). "
                            "Separate restarted/resumed logging sessions before exporting."
                        )
                    duplicate_count += 1
                    continue
                points[tag][row["timesteps"]] = row
    missing = [tag for tag in TAGS if not points[tag]]
    if missing:
        raise ValueError(f"Missing scalar tags in {run_dir.name}: {', '.join(missing)}")
    rows = {
        tag: [by_step[step] for step in sorted(by_step)]
        for tag, by_step in points.items()
    }
    metadata = dict(
        run=run_dir.name, run_dir=str(run_dir),
        config_path=str(run_dir / "config.yaml"),
        event_files=[str(f.relative_to(run_dir)) for f in files],
        identical_duplicates_collapsed=duplicate_count,
        metrics={
            tag: dict(
                count=len(values), first_timestep=values[0]["timesteps"],
                last_timestep=values[-1]["timesteps"],
            )
            for tag, values in rows.items()
        },
    )
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Use UTF-8 BOM for direct opening of the exported CSV in Excel."""
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--outputs-dir", type=Path, default=REPO_ROOT / "outputs",
        help="Training outputs root (default: repository outputs/).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="New directory for CSVs; must not exist. Relative paths use the current directory.",
    )
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                        help="Run directory basenames; default: the 21 screenshot runs.")
    args = parser.parse_args()
    outputs_dir = args.outputs_dir.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        parser.error(f"Destination already exists; use a new directory: {destination}")
    if len(set(args.runs)) != len(args.runs):
        parser.error("--runs contains duplicate names")
    for name in args.runs:
        if name in (".", "..") or "/" in name or "\\" in name or Path(name).name != name:
            parser.error(f"Expected a run basename, got: {name!r}")
    run_dirs = [outputs_dir / name for name in args.runs]
    missing = [str(p) for p in run_dirs if not p.is_dir()]
    if missing:
        parser.error("Missing run directories:\n" + "\n".join(missing))
    # Validate every input before creating the export destination.
    collected = []
    for run_dir in run_dirs:
        try:
            rows, metadata = read_run(run_dir)
        except ImportError as exc:
            parser.error(f"Missing dependency: {exc}. Activate the project's Python environment.")
        except Exception as exc:
            parser.error(f"{run_dir.name}: {exc}")
        collected.append((rows, metadata))
        print(f"[read] {run_dir.name}: eval={len(rows[TAGS[0]])}, "
              f"rollout={len(rows[TAGS[1]])}")
    if any(destination == p or p in destination.parents for p in run_dirs):
        parser.error("Choose an export directory outside the selected training run directories.")
    destination.mkdir(parents=True, exist_ok=False)
    combined: list[dict[str, Any]] = []
    for rows, metadata in collected:
        folder = destination / metadata["run"]
        folder.mkdir()
        for tag in TAGS:
            write_csv(folder / (tag.replace("/", "_") + ".csv"), rows[tag])
            combined.extend(rows[tag])
    write_csv(destination / "all_success_rates.csv", combined)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        outputs_dir=str(outputs_dir), output_dir=str(destination),
        tags=list(TAGS), success_rate_unit="fraction in [0, 1]; multiply by 100 for percent",
        timesteps="Original TensorBoard global_step (total environment interactions).",
        processing="Raw logged scalars; no smoothing, interpolation, averaging, or evaluation.",
        duplicate_policy="Collapse identical tag/step/value; reject conflicting values.",
        runs=[metadata for _, metadata in collected],
    )
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    (destination / "README.txt").write_text(
        "all_success_rates.csv: long-format table for every run and both metrics.\n"
        "Each run folder contains eval_success_rate.csv and rollout_success_rate.csv.\n"
        "success_rate is a fraction [0,1], not a percentage. timesteps is the original global_step.\n"
        "eval and rollout can have different recording intervals; rows are not aligned/interpolated.\n"
        "wall_time is a Unix timestamp. source_file identifies the original event file.\n"
        "No smoothing or seed averaging was applied. No models/environments were executed.\n"
        "These are training-time logged curves, not independent final nominal/DR evaluations.\n"
        "See manifest.json and each original config.yaml for provenance and evaluation settings.\n",
        encoding="utf-8",
    )
    print(f"[done] {len(collected)} runs, {len(combined)} rows -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
