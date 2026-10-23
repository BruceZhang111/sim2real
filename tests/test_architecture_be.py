"""Orchestration regressions; never launch training."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import run_architecture_be as batch


def test_exact_order_names_seeds_and_same_interpreter():
    paths = {g: Path(f"/tmp/{g}.yaml") for g in batch.GROUPS}
    commands = batch.build_commands(paths, "unique")
    assert len(commands) == 12
    assert [c[-1] for c in commands] == list(batch.LABELS)
    for command, label in zip(commands, batch.LABELS):
        assert command[0] == batch.sys.executable
        assert command[command.index("--seed") + 1] == label[1:]
        assert command[command.index("--config") + 1] == str(paths[label[0]])
        assert command[command.index("--run-name") + 1] == "ppo_reach_unique"
        assert "--init-from" not in command


def test_start_at_omits_completed_trials():
    paths = {g: Path(f"{g}.yaml") for g in batch.GROUPS}
    commands = batch.build_commands(paths, "unique", "D2")
    assert [c[-1] for c in commands] == ["D2", "D3", "E1", "E2", "E3"]


def test_dry_run_no_writes_or_training():
    with patch.object(batch.Path, "mkdir") as mkdir,          patch.object(batch.Config, "to_yaml") as save,          patch.object(batch, "run_experiments") as run:
        assert batch.main(["--dry-run"]) == 0
    mkdir.assert_not_called()
    save.assert_not_called()
    run.assert_not_called()


def test_real_configs_validate():
    assert tuple(batch.load_configs()) == ("B", "C", "D", "E")


def test_reject_drifted_recipe():
    configs = batch.load_configs()
    configs["B"].train.n_envs = 18
    with patch.object(batch.Config, "from_yaml", return_value=configs["B"]):
        with pytest.raises(ValueError, match="8env"):
            batch.load_configs()


def test_execution_freezes_configs_before_dispatch(tmp_path):
    configs = batch.load_configs()
    with patch.object(batch, "REPOSITORY", tmp_path),          patch.object(batch, "load_configs", return_value=configs),          patch.object(batch, "run_experiments") as run:
        run.return_value = 7
        assert batch.main([]) == 7
    commands = run.call_args.args[0]
    assert len(commands) == 12
    for command in commands:
        path = Path(command[command.index("--config") + 1])
        assert path.is_file()
        frozen = batch.Config.from_yaml(path)
        assert frozen.train.n_envs == 8 and frozen.train.target_kl == 0.33
    plans = list((tmp_path / "outputs").glob("*/commands.json"))
    assert len(plans) == 1
    assert json.loads(plans[0].read_text()) == commands
