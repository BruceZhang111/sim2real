# Controlled PPO architecture matrix: 8 environments, target KL 0.33

All configs use revised_v2, unchanged DR/VecNormalize, 2,000,000 requested steps,
8 environments, 512 steps/environment, batch size 1024, 18 epochs, lr 3e-4,
target_kl 0.33. policy_arch is explicit in each YAML.

| Config | Actual total parameters | Training seeds |
| --- | ---: | --- |
| mlp256.yaml (baseline) | 146,955 | 0, 1, 2 |
| mlp1440.yaml | 4,236,491 | 0, 1, 2 |
| simba64_512.yaml | 4,252,235 | 0, 1, 2 |
| simba256_512.yaml | 4,751,627 | 0, 1, 2 |
| simba64_256.yaml | 1,095,243 | 0, 1, 2 |

Counts include both heads and log_std, instantiated on CPU with the real
26-dimensional observation / 5-dimensional action spaces. MLP1440 differs from
SimBa64/512 by about 0.37% in total capacity; branch allocations still differ.

Use all three CLI training seeds. The YAML default seed0 is not three runs.
No audited historical run meets both n_envs=8 and target_kl=0.33; all 15
runs must be newly trained. Historical models/config snapshots remain intact.
Origin is excluded from this experiment plan. Requested total is 30M steps;
PPO finishes full rollouts (2,002,944 actual steps/run with this recipe).

After reviewing/committing locally, pushing server main and pulling ff-only
on an idle server, a single training command is:

```bash
.venv/bin/python -m sim2real.train   --config configs/architecture_8env/mlp256.yaml   --seed 0 --run-name arch8_kl033_mlp256_seed0_UNIQUE_BATCH
```

Replace config, seed and UNIQUE_BATCH for each run. Do not reuse an existing run
directory: train currently does not reject it. This document does not launch jobs.
Evaluate every final nominal/DR for 200 episodes with eval_seed=20000 as described
in docs/EVALUATION.md. The resulting controlled matrix has 30 evaluation calls,
6000 episodes. Keep historical 8-run evaluations separate (16 calls/3200 episodes).
