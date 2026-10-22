"""Focused regressions for seeds, immutable exports and normalization pairing."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from sim2real.config import Config
from sim2real.eval import evaluate, _find_model, _normalization_path, file_hash
from sim2real.train import NormalizedEvalCallback


def test_strict_model_and_pairing(tmp_path):
    cfg = Config()
    (tmp_path / "final_model.zip").touch()
    (tmp_path / "vecnormalize.pkl").touch()
    assert _find_model(tmp_path, "final").name == "final_model.zip"
    with pytest.raises(FileNotFoundError):
        _find_model(tmp_path, "best")
    with pytest.raises(FileNotFoundError):
        _normalization_path(tmp_path, "best", cfg)
    (tmp_path / "best_model").mkdir()
    (tmp_path / "best_model/vecnormalize.pkl").touch()
    with pytest.raises(FileNotFoundError, match="manifest"):
        _normalization_path(tmp_path, "best", cfg)


def test_best_callback_saves_exact_stats_and_hashes(tmp_path):
    cfg = Config()
    cfg.train.n_envs = 1
    cfg.train.eval_freq = 1
    cfg.train.n_eval_episodes = 1
    train = VecNormalize(DummyVecEnv([lambda: gym.make("Pendulum-v1", max_episode_steps=2)]))
    test = VecNormalize(DummyVecEnv([lambda: gym.make("Pendulum-v1", max_episode_steps=2)]),
                        training=False, norm_reward=False)
    try:
        model = PPO("MlpPolicy", train, n_steps=2, batch_size=2, device="cpu")
        model.set_logger(configure(str(tmp_path / "logs"), []))
        cb = NormalizedEvalCallback.make(cfg, test, str(tmp_path / "best"), str(tmp_path / "eval"))
        cb.init_callback(model)
        cb.on_step()
        pair = json.loads((tmp_path / "best/best_pair.json").read_text())
        assert pair["model_sha256"] == file_hash(tmp_path / "best/best_model.zip")
        assert pair["normalization_sha256"] == file_hash(tmp_path / "best/vecnormalize.pkl")
        saved = VecNormalize.load(str(tmp_path / "best/vecnormalize.pkl"), test.venv)
        np.testing.assert_array_equal(saved.obs_rms.mean, train.obs_rms.mean)
        assert saved.obs_rms.count == train.obs_rms.count
    finally:
        train.close()
        test.close()


@pytest.mark.parametrize("randomize", [False, True])
def test_seeded_real_rollout_export(tmp_path, randomize):
    # Two full resets and terminal metrics exercise real MuJoCo and DR wrappers.
    from sim2real.config import REPO_ROOT
    run = REPO_ROOT / "outputs/ppo_reach_20261013_120248-revised_v2"
    if not (run / "final_model.zip").is_file():
        pytest.skip("optional local historical artifact integration test")
    stats_before = file_hash(run / "vecnormalize.pkl")
    a, b = tmp_path / "a", tmp_path / "b"
    evaluate(run, 2, randomize, eval_seed=20000, output=a)
    evaluate(run, 2, randomize, eval_seed=20000, output=b)
    x, y = [json.loads((p / "results.json").read_text()) for p in (a, b)]
    assert x["episodes"] == y["episodes"]
    assert x["summary"] == y["summary"]
    assert [r["seed"] for r in x["episodes"]] == [20000, 20001]
    assert len(list(csv.DictReader((a / "episodes.csv").open()))) == 2
    assert stats_before == file_hash(run / "vecnormalize.pkl")
    with pytest.raises(FileExistsError):
        evaluate(run, 1, randomize, output=a)


@pytest.mark.parametrize("episodes,seed", [(0, 20000), (1, -1), (2, 2**32-1)])
def test_invalid_eval_inputs(episodes, seed):
    with pytest.raises(ValueError):
        evaluate(Path("/unused"), episodes, False, eval_seed=seed)


def test_pair_tamper_rejected_before_loading_model(tmp_path):
    cfg = Config()
    cfg.to_yaml(tmp_path / "config.yaml")
    (tmp_path / "final_model.zip").write_bytes(b"model")
    (tmp_path / "vecnormalize.pkl").write_bytes(b"stats")
    (tmp_path / "final_pair.json").write_text(json.dumps(dict(
        model_sha256="incorrect", normalization_sha256="incorrect")))
    with pytest.raises(ValueError, match="pair hash mismatch"):
        evaluate(tmp_path, 1, False, output=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


def test_unnormalized_run_needs_no_stats(tmp_path):
    cfg = Config()
    cfg.train.normalize_obs = False
    cfg.train.normalize_reward = False
    assert _normalization_path(tmp_path, "final", cfg) is None
