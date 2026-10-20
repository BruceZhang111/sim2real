"""Tests for the paper-faithful SimBa backbone and its SB3 integration."""

from __future__ import annotations

import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from torch import nn

from sim2real.config import Config, REPO_ROOT
from sim2real.policies import SimBaActorCriticPolicy, SimBaBlock, SimBaMlpExtractor
from sim2real.train import build_model


def test_simba_extractor_matches_paper_ppo_architecture():
    extractor = SimBaMlpExtractor(feature_dim=26, device="cpu")

    assert extractor.latent_dim_pi == 256
    assert extractor.latent_dim_vf == 512
    assert isinstance(extractor.policy_net[0], nn.Linear)
    assert extractor.policy_net[0].in_features == 26
    assert extractor.policy_net[0].out_features == 256
    assert sum(isinstance(layer, SimBaBlock) for layer in extractor.policy_net) == 1
    assert isinstance(extractor.value_net[0], nn.Linear)
    assert extractor.value_net[0].in_features == 26
    assert extractor.value_net[0].out_features == 512
    assert sum(isinstance(layer, SimBaBlock) for layer in extractor.value_net) == 2

    actor_block = next(layer for layer in extractor.policy_net if isinstance(layer, SimBaBlock))
    critic_block = next(layer for layer in extractor.value_net if isinstance(layer, SimBaBlock))
    assert (actor_block.mlp[0].in_features, actor_block.mlp[0].out_features) == (256, 1024)
    assert (actor_block.mlp[2].in_features, actor_block.mlp[2].out_features) == (1024, 256)
    assert (critic_block.mlp[0].in_features, critic_block.mlp[0].out_features) == (512, 2048)
    assert (critic_block.mlp[2].in_features, critic_block.mlp[2].out_features) == (2048, 512)
    assert isinstance(extractor.policy_net[-1], nn.LayerNorm)
    assert isinstance(extractor.value_net[-1], nn.LayerNorm)
    assert extractor.policy_net[-1].eps == 1e-6
    assert extractor.value_net[-1].eps == 1e-6


def test_reach_config_exposes_paper_simba_defaults():
    cfg = Config.from_yaml(REPO_ROOT / "configs/reach.yaml")
    assert cfg.train.policy_arch == "mlp"
    assert cfg.train.simba_actor_hidden_dim == 256
    assert cfg.train.simba_actor_n_blocks == 1
    assert cfg.train.simba_critic_hidden_dim == 512
    assert cfg.train.simba_critic_n_blocks == 2


def test_simba_extractor_shapes_and_gradients_are_finite():
    extractor = SimBaMlpExtractor(feature_dim=26, device="cpu")
    features = th.randn(7, 26, requires_grad=True)

    latent_pi, latent_vf = extractor(features)
    assert latent_pi.shape == (7, 256)
    assert latent_vf.shape == (7, 512)

    (latent_pi.square().mean() + latent_vf.square().mean()).backward()
    assert features.grad is not None
    assert th.isfinite(features.grad).all()
    assert all(
        parameter.grad is not None and th.isfinite(parameter.grad).all()
        for parameter in extractor.parameters()
    )


def test_simba_policy_preserves_sb3_actor_critic_contract():
    policy = SimBaActorCriticPolicy(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
        lr_schedule=lambda _: 3e-4,
        net_arch=[],
    )
    observations = th.randn(4, 26)

    actions, values, log_prob = policy(observations)
    eval_values, eval_log_prob, entropy = policy.evaluate_actions(observations, actions)
    predicted_values = policy.predict_values(observations)

    assert actions.shape == (4, 5)
    assert values.shape == eval_values.shape == predicted_values.shape == (4, 1)
    assert log_prob.shape == eval_log_prob.shape == entropy.shape == (4,)
    assert all(
        th.isfinite(tensor).all()
        for tensor in (actions, values, log_prob, eval_values, eval_log_prob, entropy, predicted_values)
    )
    predicted_actions, _ = policy.predict(observations.numpy(), deterministic=True)
    assert np.all(predicted_actions >= -1.0) and np.all(predicted_actions <= 1.0)
    assert sum(parameter.numel() for parameter in policy.parameters()) == 4_751_627


def test_simba_policy_save_load_preserves_custom_architecture(tmp_path):
    policy = SimBaActorCriticPolicy(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        lr_schedule=lambda _: 3e-4,
        net_arch=[],
        simba_actor_hidden_dim=32,
        simba_actor_n_blocks=2,
        simba_critic_hidden_dim=48,
        simba_critic_n_blocks=3,
    )
    observations = np.random.default_rng(0).normal(size=(3, 6)).astype(np.float32)
    action_before, _ = policy.predict(observations, deterministic=True)
    policy_path = tmp_path / "simba_policy"
    policy.save(policy_path)

    loaded = SimBaActorCriticPolicy.load(policy_path, device="cpu")
    action_after, _ = loaded.predict(observations, deterministic=True)
    assert loaded.simba_actor_hidden_dim == 32
    assert loaded.simba_actor_n_blocks == 2
    assert loaded.simba_critic_hidden_dim == 48
    assert loaded.simba_critic_n_blocks == 3
    np.testing.assert_allclose(action_after, action_before, rtol=0, atol=0)


def test_build_model_selects_simba_without_network_side_normalizer(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = Config()
    cfg.train.n_envs = 1
    cfg.train.n_steps = 8
    cfg.train.batch_size = 8
    cfg.train.n_epochs = 1
    cfg.train.device = "cpu"
    cfg.train.policy_arch = "simba"

    env = DummyVecEnv([lambda: gym.make("Pendulum-v1", max_episode_steps=8)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    try:
        model = build_model(cfg, env, str(tmp_path))
        assert isinstance(model.policy, SimBaActorCriticPolicy)
        assert not any("running" in type(module).__name__.lower() for module in model.policy.modules())
        model.learn(total_timesteps=8)

        observation = np.array([[0.2, -0.3, 0.4]], dtype=np.float32)
        action_before, _ = model.predict(observation, deterministic=True)
        model_path = tmp_path / "simba_model"
        model.save(model_path)

        from stable_baselines3 import PPO

        loaded = PPO.load(model_path, device="cpu")
        action_after, _ = loaded.predict(observation, deterministic=True)
        assert isinstance(loaded.policy, SimBaActorCriticPolicy)
        np.testing.assert_allclose(action_after, action_before, rtol=0, atol=0)

        load_code = (
            "from stable_baselines3 import PPO\n"
            f"model = PPO.load({str(model_path.with_suffix('.zip'))!r}, device='cpu')\n"
            "print(type(model.policy).__module__, type(model.policy).__name__)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", load_code],
            cwd="/tmp",
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == (
            "sim2real.policies.simba SimBaActorCriticPolicy"
        )
    finally:
        env.close()


def test_simba_runs_one_so101_ppo_update(tmp_path):
    from sim2real.env_factory import make_training_venv

    cfg = Config()
    cfg.dr.enabled = False
    cfg.train.policy_arch = "simba"
    cfg.train.device = "cpu"
    cfg.train.n_envs = 1
    cfg.train.n_steps = 8
    cfg.train.batch_size = 8
    cfg.train.n_epochs = 1
    cfg.train.normalize_reward = False

    env = make_training_venv(cfg)
    try:
        model = build_model(cfg, env, str(tmp_path))
        assert model.observation_space.shape == (26,)
        assert model.action_space.shape == (5,)
        model.learn(total_timesteps=8)
        assert model.num_timesteps == 8
    finally:
        env.close()


def test_simba_actor_exports_to_onnx_with_runtime_parity(tmp_path):
    import onnxruntime as ort

    from sim2real.export_policy import OnnxableActor

    policy = SimBaActorCriticPolicy(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
        lr_schedule=lambda _: 3e-4,
        net_arch=[],
    )
    wrapper = OnnxableActor(
        policy,
        "ppo",
        np.zeros(26, dtype=np.float32),
        np.ones(26, dtype=np.float32),
        10.0,
        1e-8,
    ).eval()
    onnx_path = tmp_path / "simba_policy.onnx"
    th.onnx.export(
        wrapper,
        th.zeros(1, 26),
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    observations = np.random.default_rng(0).normal(size=(3, 26)).astype(np.float32)
    with th.no_grad():
        torch_actions = wrapper(th.from_numpy(observations)).numpy()
    onnx_actions = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(["action"], {"obs": observations})[0]
    np.testing.assert_allclose(onnx_actions, torch_actions, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("algo", "normalize_obs", "message"),
    [
        ("sac", True, "supported only for PPO"),
        ("ppo", False, "requires train.normalize_obs=true"),
    ],
)
def test_build_model_rejects_invalid_simba_combinations(
    algo, normalize_obs, message, tmp_path
):
    cfg = Config()
    cfg.train.algo = algo
    cfg.train.policy_arch = "simba"
    cfg.train.normalize_obs = normalize_obs
    with pytest.raises(ValueError, match=message):
        build_model(cfg, None, str(tmp_path))


def test_build_model_rejects_simba_without_vec_normalize(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = Config()
    cfg.train.policy_arch = "simba"
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    try:
        with pytest.raises(ValueError, match="requires a VecNormalize environment"):
            build_model(cfg, env, str(tmp_path))
    finally:
        env.close()


def test_build_model_keeps_default_mlp_policy(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = Config()
    cfg.train.device = "cpu"
    cfg.train.n_envs = 1
    cfg.train.n_steps = 8
    cfg.train.batch_size = 8
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    try:
        model = build_model(cfg, env, str(tmp_path))
        assert not isinstance(model.policy, SimBaActorCriticPolicy)
        assert model.policy.net_arch == [256, 256]
    finally:
        env.close()


def test_build_model_rejects_unknown_policy_architecture(tmp_path):
    cfg = Config()
    cfg.train.policy_arch = "unknown"
    with pytest.raises(ValueError, match="unknown policy architecture"):
        build_model(cfg, None, str(tmp_path))
