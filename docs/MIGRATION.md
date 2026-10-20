# Local Git -> SSH -> Linux training server

## Current choice (2026-10-20)
The user opted out of proxy forwarding and server GitHub access.
All source/config/dependency edits stay in local WSL:
 /home/brucezhang/Dev/Projects/SO101 Sim2Real/sim2real
GitHub origin remains an optional backup. The added local remote is:
 server = ssh://zyg@10.181.213.48:3389/home/zyg/git/sim2real.git
No local upstream or default push destination was changed.
Windows Remote-SSH config is distinct from WSL SSH configuration/known_hosts.
WSL Git must authenticate independently; verify host fingerprint on first use.
A successful Windows VS Code SSH login does not establish WSL key access.
Passwords are entered only in the user's terminal/UI, never in chat.

## Server directory roles
 /home/zyg/git/sim2real.git : proposed bare receiver; stores commits, no editable files
 /home/zyg/Projects/sim2real : user-approved execution checkout
The bare receiver is not created or verified yet. The training checkout may
contain residue from a failed GitHub clone; inspect before clone/reuse.
Receiving a push must not automatically update the training checkout.
No hooks, force pushes or system services are required.

## 1. User creates receiver in the server SSH terminal
Run only in the server terminal:
mkdir -p /home/zyg/git
if [ -e /home/zyg/git/sim2real.git ] || [ -L /home/zyg/git/sim2real.git ]; then
    printf '%s\n' 'STOP: target already exists; inspect it before proceeding.'
else
    git init --bare --initial-branch=main /home/zyg/git/sim2real.git
fi
If creation succeeds:
git --git-dir=/home/zyg/git/sim2real.git rev-parse --is-bare-repository
Expected: true. Do not initialize over an existing unknown repository.

## 2. User sends local committed main over SSH
First review/commit local workflow documentation in WSL; uncommitted files
are not included in any push. The remote was prepared locally by AI.
In WSL project terminal:
git push server main
Or VS Code WSL Source Control -> ... -> Push to... -> server -> main.
Do not use -u: retain existing main tracking of origin/main.
No proxy or GitHub connection is needed for this push.
If password prompts fail in GUI, use the WSL terminal for the first push.
Verify host-key fingerprint; WSL may not share Windows known_hosts.
If a push is rejected, inspect history; do not force it.

## 3. Server clones from the local receiver after first push
Check destination before creating/reusing it:
ls -ld /home/zyg/Projects/sim2real
If it does not exist (or is verified empty), create parent then clone:
mkdir -p /home/zyg/Projects
git clone --no-hardlinks --branch main /home/zyg/git/sim2real.git /home/zyg/Projects/sim2real
For GUI: Git: Clone -> /home/zyg/git/sim2real.git -> select /home/zyg/Projects.
If destination contains files, stop and inspect; do not rm/reset/clean it.
Open /home/zyg/Projects/sim2real in the SSH window.
Run:
git config --local pull.ff only
git status --short --branch
git rev-parse HEAD
Expected origin is the SERVER FILESYSTEM bare path, not GitHub.
Compare local WSL and server full HEAD SHA.

## 4. Daily workflow
LOCAL WSL: AI edit -> checks -> review -> Commit -> Push to server main.
Optional separate Push to origin main for GitHub backup.
SERVER: after training is stopped/finished and checkout clean:
git pull --ff-only origin main
Compare SHA, then sync the validated locked uv environment and start next run.
Do not edit source on the server; report errors/logs to local AI.
Do not pull/sync environment in a checkout used by running training.
A push may reach the separate bare receiver during training, but checkout
update must wait. Avoid hooks that deploy automatically.
Do not configure receive.denyCurrentBranch=ignore or use force to bypass errors.

## Proxy removal (user side)
Remove RemoteForward 127.0.0.1:17890 127.0.0.1:7890 from the local Windows SSH
Host block, then disconnect/reconnect that SSH connection.
If the GitHub proxy was set on server:
git config --global --unset-all http.https://github.com.proxy
Unsetting a nonexistent key may exit nonzero without output.
No command above is executed on server by AI.
Shared loopback does not mean per-user authentication; no proxy is required
for this direct Git-over-SSH workflow.

## Environment and artifacts
SSH Git transfer solves source sync only, not package downloads.
uv, Python, PyTorch and other wheel availability on server must be checked.
If downloads are blocked, plan local download plus artifact transfer/offline
installation. Do not silently restore the rejected proxy.
Server: Ubuntu 22.04.2 x86_64, driver 525.147.05, 2 RTX 4090, both busy at last
user snapshot; 36 cores/72 threads, ~251 GiB RAM; Python 3.10.12; uv absent in PATH.
Target project Python remains 3.11. Local torch 2.14.0+cu130 is not an appropriate
default for this driver. Do not change shared driver or claim environment parity.
No uv lock has been created yet. Prepare compatible metadata/lock locally,
then transfer with Git. System Python and local conda remain unchanged.
Existing project dev dependencies are an extra (--extra dev).
Source XML/STL assets are tracked; outputs (~15 GB), models and environments
are ignored and require separate transfer if needed, with matched normalization.
Never use make clean as a migration step. Choose GPU allocation/scheduler policy
and persistent execution before formal training.

## Current evidence
Local tree was clean at 8c74c201fd734d1a42ca643945a41502d097fd40 before this change.
User verified GitHub via proxy at that commit, then rejected proxy dependency.
Latest 30-second direct GitHub test produced no output; exit status was not
captured, so timeout is suspected, not proven. No bare init, push or clone
success has been reported yet. Local server remote setup is complete only.

## Direct SSH checkout verified by user (2026-10-20)

User supplied: execution cwd /home/zyg/Projects/sim2real, clean main tracking origin/main, origin /home/zyg/git/sim2real.git, HEAD b85adacdc2bb31c7696044703ceb709a7198b058 matching local WSL. Fast-forward-only pull configuration command completed without reported error. Direct SSH receiver/push/checkout workflow is verified by user outputs; AI has not logged into the server. Local SB3 2.9.0 installed metadata requires torch>=2.8,<3.0. Therefore selecting an older CUDA 11.8 torch build alone may conflict; do not silently downgrade SB3 or assert equivalence. Next: install user-scoped uv through an available download route, verify package index reachability, then resolve and test a driver-compatible stack locally. uv.lock and remote Python3.11 environment are not yet created.

## uv bootstrap without server pip (2026-10-20)

User reports /usr/bin/python3 has no pip. Prepared official PyPI uv 0.13.3 manylinux x86_64 wheel locally; verified wheel SHA256 against PyPI and extracted the standalone uv binary. Local --version succeeded. Ignored transfer files are under outputs/migration-tools/uv-0.13.3-linux-x86_64/; provenance in docs/environment/server-20261020/uv-bootstrap.json. User must transfer binary/checksum over SSH, validate and install in ~/.local/bin. No server install has been performed. No system pip or sudo required. Python3.11 download and project dependencies remain pending.
