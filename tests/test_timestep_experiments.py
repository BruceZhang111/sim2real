"""Lightweight orchestration tests: never start PPO or a simulator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run_timestep_experiments as sweep
from sim2real.config import Config


class TimestepExperimentsTest(unittest.TestCase):
    def test_shared_config_and_only_budget_name_differ(self):
        cfg = Config()
        cfg.train.policy_arch = "simba"
        commands = sweep.build_commands(cfg, Path("/tmp/frozen config.yaml"))
        self.assertEqual(len(commands), 3)
        for command, budget in zip(commands, sweep.TIMESTEP_BUDGETS):
            self.assertEqual(command[:7], commands[0][:7])
            self.assertEqual(command[-3], str(budget))
            self.assertEqual(command[-1], f"ppo_reach-simba-ts{budget}")
            self.assertNotIn("--init-from", command)

    def test_loader_preserves_environment_and_training_options(self):
        cfg = Config()
        cfg.train.n_envs = 18
        cfg.train.n_epochs = 18
        cfg.train.target_kl = 0.33
        cfg.train.run_name = "old-name"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.yaml"
            cfg.to_yaml(source)
            loaded = sweep.load_config(source, "simba")
        self.assertEqual(loaded.env, cfg.env)
        self.assertEqual(loaded.dr, cfg.dr)
        expected = cfg.to_dict()["train"]
        expected.update(policy_arch="simba", run_name=None)
        self.assertEqual(loaded.to_dict()["train"], expected)

    def test_reject_warm_start(self):
        cfg = Config()
        cfg.train.init_from = "old.zip"
        with patch.object(sweep.Config, "from_yaml", return_value=cfg):
            with self.assertRaisesRegex(ValueError, "from scratch"):
                sweep.load_config(Path("unused"), "simba")

    def test_dry_run_never_writes_or_starts_training(self):
        with patch.object(sweep.Config, "from_yaml", return_value=Config()), \
             patch.object(sweep.Config, "to_yaml") as save, \
             patch.object(sweep.Path, "mkdir") as mkdir, \
             patch.object(sweep.subprocess, "Popen") as popen:
            self.assertEqual(sweep.main(["--dry-run"]), 0)
        save.assert_not_called()
        mkdir.assert_not_called()
        popen.assert_not_called()

    def test_success_runs_serially(self):
        commands = [["python", str(budget)] for budget in sweep.TIMESTEP_BUDGETS]
        events = []

        def start(command, **kwargs):
            events.append(("start", command))
            process = Mock()
            process.wait.side_effect = lambda: events.append(("wait", command)) or 0
            return process

        with patch.object(sweep.subprocess, "Popen", side_effect=start):
            self.assertEqual(sweep.run_experiments(commands), 0)
        self.assertEqual(events, [event for cmd in commands
                                  for event in (("start", cmd), ("wait", cmd))])

    def test_main_freezes_config_before_all_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config()
            cfg.train.log_dir = str(Path(directory) / "outputs")
            cfg.train.n_envs = 18
            cfg.train.n_epochs = 18
            cfg.train.target_kl = 0.33
            source = Path(directory) / "input.yaml"
            cfg.to_yaml(source)
            seen = []

            def start(command, **kwargs):
                snapshot = Path(command[command.index("--config") + 1])
                frozen = Config.from_yaml(snapshot)
                self.assertEqual(frozen.env, cfg.env)
                self.assertEqual(frozen.dr, cfg.dr)
                self.assertEqual(frozen.train.n_envs, 18)
                self.assertEqual(frozen.train.n_epochs, 18)
                self.assertEqual(frozen.train.target_kl, 0.33)
                self.assertEqual(frozen.train.policy_arch, "simba")
                self.assertIsNone(frozen.train.init_from)
                seen.append(snapshot)
                # Editing the source between trials must not alter the snapshot.
                changed = Config()
                changed.train.n_envs = 1
                changed.to_yaml(source)
                process = Mock()
                process.wait.return_value = 0
                return process

            with patch.object(sweep.subprocess, "Popen", side_effect=start):
                self.assertEqual(sweep.main(["--config", str(source)]), 0)
            self.assertEqual(len(seen), 3)
            self.assertEqual(len(set(seen)), 1)
            self.assertTrue((seen[0].parent / "commands.txt").is_file())

    def test_failure_stops_remaining_runs(self):
        process = Mock()
        process.wait.return_value = 7
        with patch.object(sweep.subprocess, "Popen", return_value=process) as popen:
            self.assertEqual(sweep.run_experiments([["first"], ["second"]]), 7)
        self.assertEqual(popen.call_count, 1)

    def test_interrupt_waits_and_stops_remaining_runs(self):
        process = Mock()
        process.wait.side_effect = [KeyboardInterrupt, 0]
        with patch.object(sweep.subprocess, "Popen", return_value=process) as popen:
            self.assertEqual(sweep.run_experiments([["first"], ["second"]]), 130)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
