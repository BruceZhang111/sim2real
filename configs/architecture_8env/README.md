# Controlled PPO architecture matrix: 8 environments, target KL 0.33

All configs use revised_v2, unchanged DR/VecNormalize, 2,000,000 requested steps,
8 environments, 512 steps/environment, batch size 1024, 18 epochs, lr 3e-4,
target_kl 0.33. policy_arch is explicit in each YAML.

| Config | Actual total parameters | Training seeds |
| --- | ---: | --- |
| mlp256.yaml (baseline) | 146,955 | 1, 2, 3 |
| mlp1440.yaml | 4,236,491 | 1, 2, 3 |
| simba64_512.yaml | 4,252,235 | 1, 2, 3 |
| simba256_512.yaml | 4,751,627 | 1, 2, 3 |
| simba64_256.yaml | 1,095,243 | 1, 2, 3 |

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
.venv/bin/python -m sim2real.train   --config configs/architecture_8env/mlp256.yaml   --seed 3 --addname A3
```

Replace config, seed and UNIQUE_BATCH for each run. Do not reuse an existing run
directory: train currently does not reject it. This document does not launch jobs.
Evaluate every final nominal/DR for 200 episodes with eval_seed=20000 as described
in docs/EVALUATION.md. The resulting controlled matrix has 30 evaluation calls,
6000 episodes. Keep historical 8-run evaluations separate (16 calls/3200 episodes).

## Remaining B-E batch (user has completed A seeds 1/2/3)

User report: A1/A2/A3 training completed; artifacts and performance have not been
independently checked. Current training seed set is 1/2/3 for all five groups.
Do not rerun A automatically. B-E requests 12 fresh runs, 24M environment steps.

After local review/commit/push and server ff-only synchronization, run on the server:

```bash
cd /home/zyg/Projects/sim2real
source .venv/bin/activate
python scripts/run_architecture_be.py --dry-run
python scripts/run_architecture_be.py
```

Order is B1 B2 B3 C1 C2 C3 D1 D2 D3 E1 E2 E3. Each child explicitly passes
--seed and --addname (for example --seed 3 --addname B3). Outputs use a unique
timestamp/UUID batch prefix plus the group-seed suffix, e.g. ppo_reach_BATCH-B3.
Every run still uses 8env/KL0.33/2M. The batch freezes four YAML snapshots and
commands.json under outputs/architecture_be_BATCH before launching any training;
actual run outputs remain in outputs/ppo_reach_BATCH-B1 etc.

Runs are sequential; any nonzero child return or Ctrl+C stops the sequence.
A new invocation starts a new batch, without overwriting or automatically skipping
past completed work. To restart from a specific trial (fresh training, not checkpoint
continuation), for example after C2 failed:

```bash
python scripts/run_architecture_be.py --start-at C2
```

C1 and earlier are omitted in that invocation. Use tmux for SSH disconnection
protection and keep source/checkout unchanged during training. Script defaults
to the Python interpreter used to launch it; activate the intended .venv first.
