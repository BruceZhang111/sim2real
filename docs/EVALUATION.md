# Fixed final evaluation protocol (2026-10-22)

Source edits stay in local WSL. No automatic commit, push, pull or evaluation job.
Historical outputs are retained. Review and commit locally when ready, then:

```bash
# local WSL, after the user has reviewed and committed the change:
git push server main

# server, only after checking that no training uses this checkout:
cd /home/zyg/Projects/sim2real
git status --short --branch
git pull --ff-only
git rev-parse HEAD
uv pip freeze --python .venv/bin/python  # retain actual environment snapshot separately
```

The environment must already be prepared; verify the local/server code SHA,
dependencies and asset versions before interpreting comparisons. No dependency
changes or lockfile were made by this evaluation patch.

For each run below, execute these two commands on the server (substitute RUN).
Default output is a new timestamped directory under RUN/eval_raw; --output may
specify a new directory, but an existing directory is rejected.

```bash
.venv/bin/python -m sim2real.eval --run "$RUN" --which final --episodes 200 --eval-seed 20000
.venv/bin/python -m sim2real.eval --run "$RUN" --which final --episodes 200 --eval-seed 20000 --randomize
```

Audited local run names (verify server availability; do not substitute silently):

- outputs/ppo_reach_20261012_114650_nenvs8_origin (reward history only)
- outputs/ppo_reach_20261013_120248-revised_v2 (architecture baseline)
- outputs/ppo_reach_20261019_110612-simba-L12
- outputs/ppo_reach_20261019_174807-KL0.33
- outputs/ppo_reach_20261020_104421-actor128-critic512
- outputs/ppo_reach_20261020_113536-actor128-critic1024
- outputs/ppo_reach_20261020_154940-actor128-critic1024-timesteps
- outputs/ppo_reach_20261021_203622-actor64-critic512
- outputs/timestep_sweep_20261019_222032_a097e2ee/ppo_reach-simba-ts5000000

Evaluation uses deterministic CPU inference, frozen observation statistics and
unnormalized rewards. Episode i explicitly resets with seed 20000+i, independent
of training seed and previous episode duration. Within a mode the same config and
seed list reproduce the same reset target across models. Nominal/DR consume
different random draws: they are distinct test strata, not paired target scenes.
Training-time callback seed 10000+training seed is separate from this test set.
Do not choose best models or tune recipes against these held-out test results.

Every export contains metadata.json, episodes.csv, and results.json. Metadata
records model/config/normalization SHA256, code SHA, dirty status/diff hash,
dependency versions and seed protocol. results.json retains initial targets to
audit equal reset scenes. Distances are meters; lengths are control steps;
duration is seconds. Command smoothness is the episode mean of
sum((a_t-a_prev)^2), starting with a_prev=0; command magnitude is mean sum(a_t^2).
These are policy commands before latency/noise, not joint velocity or executed
actuator smoothness. Report episode-averaged summaries accordingly.

Final requires its run-root vecnormalize.pkl when normalization is configured.
Legacy final pairing is identified by directory convention and recorded hashes;
this cannot retrospectively prove the two files were saved simultaneously.
No model fallback or identity-statistics fallback occurs. Historical best without
its own statistics and best_pair.json fails explicitly. Future training saves
best_model/vecnormalize.pkl and a matching SHA256 manifest immediately after
SB3 saves a new best; no historical files are reconstructed.

The short local tests establish determinism and export correctness, not 200-episode
performance or cross-machine bitwise identity. Complete server re-evaluation before
freezing the 5 architectures x 3 training seeds; historical n_envs/KL/software
confounds and single-training-seed limitations remain. Fourier work follows the
performance evidence and does not establish causality.

Future final saves also emit final_pair.json. If this manifest exists, evaluation
checks both hashes and fails on tampering; historical final remains explicitly
labelled as a legacy run-relative pair.
