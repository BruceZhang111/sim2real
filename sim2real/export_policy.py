"""Export a trained SB3 policy to ONNX for deployment.

The observation normalization (VecNormalize running mean/var) is baked into the
ONNX graph, so the real-robot controller can feed *raw* observations and get an
action back — no SB3, no torch, no separate stats file to keep in sync. This
removes a classic sim2real footgun (forgetting to normalize at deploy time).

    python -m sim2real.export_policy --run outputs/ppo_reach_XXXX
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sim2real.config import Config, REPO_ROOT


class OnnxableActor(nn.Module):
    """Deterministic actor with normalization folded in: raw obs -> action."""

    def __init__(self, policy, algo: str, obs_mean, obs_var, clip_obs: float, eps: float):
        super().__init__()
        self.algo = algo
        self.clip_obs = float(clip_obs)
        self.eps = float(eps)
        self.register_buffer("mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("var", torch.as_tensor(obs_var, dtype=torch.float32))
        if algo == "ppo":
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net
        elif algo == "sac":
            self.actor = policy.actor
        else:
            raise ValueError(algo)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = (obs - self.mean) / torch.sqrt(self.var + self.eps)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        if self.algo == "ppo":
            latent_pi, _ = self.mlp_extractor(obs)
            return self.action_net(latent_pi)  # Gaussian mean; caller clips to [-1,1]
        # SAC deterministic action = tanh(mu(latent_pi(features)))
        features = self.actor.extract_features(obs, self.actor.features_extractor)
        latent = self.actor.latent_pi(features)
        return torch.tanh(self.actor.mu(latent))


def _algo_cls(name: str):
    from stable_baselines3 import PPO, SAC
    return {"ppo": PPO, "sac": SAC}[name]


def _load_obs_stats(run_dir: Path, obs_dim: int):
    vn_path = run_dir / "vecnormalize.pkl"
    if not vn_path.exists():
        return np.zeros(obs_dim, dtype=np.float32), np.ones(obs_dim, dtype=np.float32), 10.0, 1e-8
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    return (
        vn.obs_rms.mean.astype(np.float32),
        vn.obs_rms.var.astype(np.float32),
        float(vn.clip_obs),
        float(vn.epsilon),
    )


def export(run_dir: Path, which: str = "best") -> Path:
    cfg = Config.from_yaml(run_dir / "config.yaml")
    model_path = (run_dir / "best_model" / "best_model.zip")
    if not model_path.exists():
        model_path = run_dir / "final_model.zip"
    model = _algo_cls(cfg.train.algo).load(str(model_path), device="cpu")
    model.policy.set_training_mode(False)

    obs_dim = int(np.prod(model.observation_space.shape))
    mean, var, clip_obs, eps = _load_obs_stats(run_dir, obs_dim)

    wrapper = OnnxableActor(model.policy, cfg.train.algo, mean, var, clip_obs, eps)
    wrapper.eval()
    norm = (mean, var, clip_obs, eps)

    out_path = run_dir / "policy.onnx"
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    # Use the legacy TorchScript exporter (dynamo=False): it is well proven for
    # small SB3 MLP policies and avoids the onnxscript dependency that the newer
    # dynamo exporter pulls in.
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # sidecar metadata for the deploy controller
    meta = {
        "algo": cfg.train.algo,
        "obs_dim": obs_dim,
        "action_dim": int(model.action_space.shape[0]),
        "action_joints": cfg.env.action_joints,
        "action_mode": cfg.env.action_mode,
        # per-joint list so a faster gripper delta (gripper_action_scale)
        # survives the export; deploy broadcasts it elementwise
        "action_scale": [
            cfg.env.gripper_action_scale
            if j == "gripper" and cfg.env.gripper_action_scale is not None
            else cfg.env.action_scale
            for j in cfg.env.action_joints
        ],
        "control_freq": cfg.env.control_freq,
        "include_last_action": cfg.env.include_last_action,
    }
    (run_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2))

    _verify(model, out_path, obs_dim, norm)
    print(f"[export] wrote {out_path} and policy_meta.json")
    return out_path


def _verify(model, onnx_path: Path, obs_dim: int, norm, n: int = 8) -> None:
    """Check ONNX(raw_obs) == SB3_policy(normalized_obs).

    The ONNX graph folds normalization in, so the correct reference is the SB3
    policy fed the *normalized* observation (model.predict does NOT normalize).
    """
    import onnxruntime as ort

    mean, var, clip_obs, eps = norm
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(n):
        obs = rng.normal(size=(1, obs_dim)).astype(np.float32)
        norm_obs = np.clip((obs - mean) / np.sqrt(var + eps), -clip_obs, clip_obs).astype(np.float32)
        sb3_action, _ = model.predict(norm_obs, deterministic=True)
        onnx_action = np.clip(sess.run(["action"], {"obs": obs})[0], -1.0, 1.0)
        max_err = max(max_err, float(np.max(np.abs(sb3_action - onnx_action))))
    print(f"[export] ONNX vs SB3 max action error: {max_err:.2e}")
    if max_err > 1e-4:
        print("[export] WARNING: parity error is high; check normalization/export.")


def main() -> None:
    p = argparse.ArgumentParser(description="Export SO-101 policy to ONNX")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--which", type=str, default="best", choices=["best", "final"])
    args = p.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    export(run_dir, args.which)


if __name__ == "__main__":
    main()
