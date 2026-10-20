# Local WSL development, Git synchronization, server training

## Roles and verified state (2026-10-20)
- Sole development source: /home/brucezhang/Dev/Projects/SO101 Sim2Real/sim2real
- AI edits code, configs, dependency declarations and documentation in local WSL.
- User confirms the execution server runs Linux; its distribution, GPU, driver,
  disk, scheduler, project path and authenticated access are not yet inspected.
- Server is an execution checkout. Do not develop/commit/push source changes there.
- Existing origin: https://github.com/BruceZhang111/sim2real.git
- Existing branch: main. GitHub access from server and repository visibility
  remain unverified. No remote operation or authenticated login was performed.
- Administrator-provided SSH endpoint: 10.181.213.48:3389.
- WSL TCP probe received an Ubuntu OpenSSH banner; this is not login/identity proof.
- Local migration edits remain uncommitted. They will not be transferred by
  clone/pull until reviewed, committed and pushed.
- Tracked assets include the SO-101 XML and STL files (18 tracked asset paths).
- outputs/ is ignored, approximately 15 GB; assets approximately 16 MB.
- Local inventory is under docs/environment/wsl-20261020/. pip check passed.

## First setup
1. Connect through VS Code Remote-SSH GUI, using
   ssh -p 3389 YOUR_USER@10.181.213.48 in Add New SSH Host.
   Verify the host fingerprint; enter passwords only in the local UI.
   Keeping separate WSL and SSH windows is optional but makes context explicit.
   A Windows-hosted Remote-SSH connection does not automatically route via WSL.
2. Inspect server resources and site policies. Confirm an absolute project
   parent path with sufficient quota. Inspect OS, GPU driver, RAM/disk, uv/git,
   outbound GitHub/package access and whether a scheduler is required.
3. Feed sanitized inspection results to the local AI. AI prepares Python 3.11,
   selected PyTorch index/version, dependency constraints and uv.lock locally,
   using the captured baseline. Never fabricate a lock or claim a successful sync.
   .python-version selects Python; .venv is created independently per host.
   Preserve local conda. uv init is unnecessary for this existing pyproject.
4. In WSL VS Code Source Control, review each migration diff, stage the intended
   files, Commit, then ... -> Push to existing origin/main. Commit alone does
   not upload. Include pyproject.toml, validated uv.lock, .python-version and
   intended docs/configs. Do not stage secrets, .venv, outputs or caches.
   Resolve any unexpected upstream changes locally before pushing.
5. In the VS Code SSH window use Git: Clone, enter the existing HTTPS origin,
   select the confirmed SERVER directory, and open the resulting remote folder.
   No manual source-folder upload is needed on this normal GitHub path.
   Private repository access requires separate server-side Git authentication.
   The server login password is not a GitHub password. Prefer scoped read-only
   access (e.g. repository deploy key) under the site's policy. Do not embed
   credentials in URLs/commands, send private keys to chat, or enable broad
   agent forwarding merely to make clone work.
6. Confirm local and server HEAD match and server worktree is clean.
   On server, configure once with: git config --local pull.ff only
   This makes ordinary Pull refuse diverged history.
7. Install user-scoped uv on the server under site policy; no driver/system
   changes without an agreed plan. Only after the lock and PyTorch index are
   committed: uv sync --locked --python 3.11 --extra dev
   Existing dev is an optional extra, not a dependency group.
   Select remote .venv/bin/python in VS Code Python: Select Interpreter.
8. Validate dependency consistency, imports, GPU and headless MuJoCo.
   With make installed: uv run --locked --extra dev make smoke
   Read failures and report them to local AI; fix source locally and repeat
   commit/push/pull. Do not run make setup (conda) as the uv installation path.
   Existing full-suite success-condition test failure is a known historical
   issue; establish actual server test results rather than claiming all pass.

## Every development/training cycle
1. Local AI edits -> local checks -> user reviews.
2. Local WSL Source Control: Stage -> Commit -> Push.
3. Ensure no training is using the server checkout before updating source/env.
4. Server Source Control: Pull (not Sync Changes); or git pull --ff-only.
   If dirty/diverged/conflicted, stop and inspect; do not force reset or clean.
5. Compare full HEAD SHA; confirm clean worktree. Record SHA with run metadata.
6. uv sync --locked --python 3.11 --extra dev
   Synchronizes with committed lock, without re-resolving dependencies remotely.
7. Submit/launch training under scheduler or persistent session per site policy.
   Keep actual config, seed, CLI overrides, code SHA, lock and matching
   model/VecNormalize artifacts. Do not pull or sync this checkout mid-training.
8. Review logs remotely; return logs/results to local AI. Fix source locally.
   For simultaneous newer experiments, plan separate fixed-commit run checkouts.

## Initial training command template (do not execute yet)
After server validation, allocation and log persistence are arranged:
uv run --locked --extra dev python -m sim2real.train --config configs/reach_forSimBa.yaml --policy-arch simba --addname server-critic1024
Current YAML still says policy_arch: mlp, so the explicit override matters.
Validate hardware suitability and experimental conditions before launching.
Starting in a normal VS Code terminal does not establish disconnect persistence.
Do not copy this placeholder template as a claim that remote setup is complete.

## Artifacts and troubleshooting
Clone transfers tracked assets but not ignored outputs, environments or weights.
Training from scratch need not receive all old outputs. To evaluate/warm-start,
transfer selected complete run folders with config/model/matching statistics.
If the user needs all historical outputs mirrored, transfer the full set
separately after checking quota; do not silently discard old data.
WinSCP SFTP GUI is an option if the server enables SFTP; source can be the WSL
UNC path directly, without making a Windows project copy. Use correct checksums/
permissions for important artifacts. Preserve model/VecNormalize pairing;
best/final pairing remains a known unresolved project risk.
Large data/model artifacts should not be force-added to Git.

If GitHub is unreachable from the server, inspect the site's network policy.
An alternative is a Git bundle transferred over SFTP, followed by offline fetch.
Do not silently change workflow, publish the repository or configure a new remote.
Raw pip/conda snapshots are inventory, not universally portable installers.
uv does not provision the NVIDIA driver or all operating-system libraries.
Manual out-of-band torch installation is not the project source of truth;
configure its accelerator index in pyproject before lock/sync.

## Planned local automation
After paths, server policy and uv configuration are known, local AI may prepare
VS Code tasks for inspection, fast-forward update, locked sync and smoke.
Review these commands before use. Long training must use a persistent session
or scheduler. No task/config was created or remote command executed here.

## Authenticated server inspection supplied by user (2026-10-20)

User logged in through VS Code Remote-SSH as zyg@linux; home /home/zyg; Ubuntu 22.04.2 x86_64. NVIDIA driver 525.147.05, two GPUs with 24564 MiB each (exact models truncated), utilization 90%/99% and memory used 17399/19388 MiB at snapshot. RAM available 210 GiB; shared /home free 6.8T, quota unknown. Git/Python3 present; uv/sbatch not found in PATH. No scheduler absence or GPU allocation may be inferred. Local torch 2.14.0+cu130 is not a compatible default for this driver. Keep local environment unchanged; do not upgrade shared driver. Record in docs/environment/server-20261020/inspection.json. Next: exact GPU/CPU models, GPU allocation policy, project path/quota, GitHub access; then choose a compatible PyTorch build and validate dependency constraints before locking. Clone/push/uv installation/training remain pending.

## Server hardware and GitHub access confirmed by user (2026-10-20)

Both GPUs are RTX 4090 (24564 MiB each), currently 99%/100% utilized in the latest supplied snapshot. CPU is dual Xeon Gold 6253CL, 36 physical cores / 72 logical CPUs. System Python 3.10.12; Git 2.34.1. User's server-side git ls-remote succeeded, returning origin HEAD 4894fb73b3821369fd23aab3427fa88142af5dce. This verifies repository read access at that time, not server push access or package-index access. Choose Python 3.11 for the project via uv; preserve system Python. Next: local review/commit/push of migration preparation, remote GUI clone into a confirmed project directory, then locally prepared driver-compatible uv lock and remote validation. uv.lock does not exist yet. GPU allocation and user quota remain unknown; no training started.
