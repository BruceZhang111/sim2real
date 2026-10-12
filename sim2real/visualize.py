"""Watch a trained SO-101 policy.

Two ways to see the policy after training:

    # interactive 3D viewer (needs a desktop session / display):
    pickplace:
    python -m sim2real.visualize --run outputs/<run> --device cpu

    
    python -m sim2real.visualize --run outputs/<run>
    python -m sim2real.visualize --run outputs/<run> --randomize   # under domain rand

    # headless: render an mp4 (works over SSH / no display):
    MUJOCO_GL=egl python -m sim2real.visualize --run outputs/<run> --video reach.mp4

The interactive viewer opens the MuJoCo window running the *same* MjData the
policy steps, and shows the red target sphere so you can watch the TCP chase it,
paced to the real control rate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from sim2real.config import Config, REPO_ROOT
from sim2real.env_factory import make_single_env


def _configure_wslg_opengl() -> tuple[bool, str | None]:
    """Prefer WSLg's D3D12 renderer instead of Mesa's llvmpipe fallback.

    The variables must be set before ``mujoco.viewer`` creates its GLFW/OpenGL
    context.  Existing user choices always win.
    """
    if sys.platform != "linux" or "microsoft" not in os.uname().release.lower():
        return False, None
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, None
    if not Path("/usr/lib/wsl/lib/libd3d12.so").exists():
        return False, None

    changed = "GALLIUM_DRIVER" not in os.environ
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")

    # Prefer the discrete GPU when the WSL NVIDIA bridge is installed.  Mesa
    # treats this value as a case-insensitive adapter-name substring.
    if Path("/usr/lib/wsl/lib/libcuda.so").exists():
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

    return changed, os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME")


def _algo_cls(name: str):
    from stable_baselines3 import PPO, SAC
    return {"ppo": PPO, "sac": SAC}[name]


def _find_model(run_dir: Path, which: str) -> Path:
    for key in ([which] + ["best", "final"]):
        p = {"best": run_dir / "best_model" / "best_model.zip",
             "final": run_dir / "final_model.zip"}[key]
        if p.exists():
            return p
    raise FileNotFoundError(f"no model found under {run_dir}")


def _unwrap_base(venv):
    v = venv
    while hasattr(v, "venv"):
        v = v.venv
    return v.envs[0].unwrapped


def watch(
    run_dir: Path,
    episodes: int,
    randomize: bool,
    which: str,
    realtime: bool = True,
    seed: int | None = None,
    episode: int | None = None,
    render_fps: float = 60.0,
) -> None:
    """Open the interactive viewer and roll out the policy.

    Episode selection:
      * ``seed`` set        -> reproducible sequence; episode k uses seed ``seed+k``.
      * ``episode`` set     -> lock onto that one scenario and replay it (implies
                               a base seed of 0 if ``seed`` is None).
      * neither             -> fresh random episodes each run.
    """
    wslg_backend_changed, wslg_adapter = _configure_wslg_opengl()
    import mujoco.viewer
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = Config.from_yaml(run_dir / "config.yaml")
    venv = DummyVecEnv([make_single_env(cfg, randomize=randomize)])
    vn_path = run_dir / "vecnormalize.pkl"
    if vn_path.exists():
        venv = VecNormalize.load(str(vn_path), venv)
        venv.training = False
        venv.norm_reward = False

    model = _algo_cls(cfg.train.algo).load(str(_find_model(run_dir, which)), device="cpu")
    base = _unwrap_base(venv)
    import inspect

    print(f"[debug] visualize file: {Path(__file__).resolve()}")
    print(f"[debug] base class file: {inspect.getfile(type(base))}")
    print(f"[debug] base class: {type(base).__name__}")

    sim_dt = float(base.model.opt.timestep)
    control_dt = float(base.frame_skip) * sim_dt
    actual_control_freq = 1.0 / control_dt

    if render_fps <= 0:
        raise ValueError("render_fps must be greater than zero")

    # 渲染频率没有必要高于物理状态产生频率
    render_fps = min(float(render_fps), 1.0 / sim_dt)
    render_dt = 1.0 / render_fps

    base_seed = 0 if (episode is not None and seed is None) else seed

    def ep_seed(k: int):
        if episode is not None:
            return base_seed + episode          # always the same scenario
        if base_seed is not None:
            return base_seed + k                # reproducible sequence
        return None                             # random

    def seeded_reset(k: int):
        s = ep_seed(k)
        if s is not None:
            venv.seed(s)                        # controls the next reset's scenario
        return venv.reset(), s

    label = (f"episode {episode} (looped)" if episode is not None
             else f"seeded from {base_seed}" if base_seed is not None else "random episodes")
    print(f"[watch] {cfg.train.algo} policy | {label} | randomize={randomize} | close the window to stop")
    print(
        f"[timing] physics={1.0 / sim_dt:.1f} Hz | "
        f"frame_skip={base.frame_skip} | "
        f"configured_control={cfg.env.control_freq:.2f} Hz | "
        f"actual_control={actual_control_freq:.2f} Hz | "
        f"render_target={render_fps:.1f} Hz"
    )
    if os.environ.get("GALLIUM_DRIVER") == "d3d12":
        source = "auto" if wslg_backend_changed else "environment"
        adapter = f" | adapter_hint={wslg_adapter}" if wslg_adapter else ""
        print(f"[graphics] WSLg backend=d3d12 ({source}){adapter}")

    obs, s = seeded_reset(0)
    ep, ep_ret = 0, 0.0

    with mujoco.viewer.launch_passive(
        base.model,
        base.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # 有些旧版 MuJoCo 不支持 state_only 参数。
        # 第一次尝试失败后，后续自动退回完整 sync。
        state_only_supported = True

        wall_origin = 0.0
        sim_origin = 0.0
        next_render_sim_time = 0.0
        sync_count = 0
        sync_report_wall = 0.0

        def sync_dynamic_state() -> None:
            """Synchronize the changing physics state to the passive viewer."""
            nonlocal state_only_supported

            if state_only_supported:
                try:
                    viewer.sync(state_only=True)
                    return
                except TypeError:
                    # 旧版 MuJoCo：viewer.sync() 没有 state_only 参数
                    state_only_supported = False

            viewer.sync()

        def reset_render_clock() -> None:
            """Restart wall-clock pacing after every environment reset."""
            nonlocal wall_origin, sim_origin, next_render_sim_time
            nonlocal sync_count, sync_report_wall

            # reset/domain randomization 可能修改 MjModel 或目标位置。
            # episode 开始时执行一次完整同步。
            viewer.sync()

            sim_origin = float(base.data.time)
            wall_origin = time.perf_counter()
            sync_report_wall = wall_origin
            sync_count = 0

            # 初始状态已经通过上面的完整 sync 显示，
            # 第一张动态帧安排在 1/render_fps 秒之后。
            next_render_sim_time = sim_origin + render_dt

        def on_physics_substep(sim_time: float) -> None:
            """Render selected intermediate MuJoCo states."""
            nonlocal next_render_sim_time, sync_count, sync_report_wall

            if not viewer.is_running():
                return

            # 还没有到下一渲染帧对应的仿真时间
            if sim_time + 1e-12 < next_render_sim_time:
                return

            if realtime:
                # 用累计仿真时间对齐墙钟时间。
                # model.predict、VecNormalize 和 env.step 的耗时都被包含在内。
                deadline = wall_origin + (sim_time - sim_origin)
                remaining = deadline - time.perf_counter()

                if remaining > 0:
                    time.sleep(remaining)

            sync_dynamic_state()
            sync_count += 1

            now = time.perf_counter()
            report_elapsed = now - sync_report_wall
            if report_elapsed >= 2.0:
                print(
                    f"[timing] viewer_sync={sync_count / report_elapsed:.1f} Hz "
                    f"(target={render_fps:.1f} Hz)"
                )
                sync_count = 0
                sync_report_wall = now

            # sim_dt 通常无法整除 render_dt。
            # 使用 while 可以跳过已经落后的渲染时刻，避免时间漂移。
            while next_render_sim_time <= sim_time + 1e-12:
                next_render_sim_time += render_dt

        base.set_physics_step_callback(on_physics_substep)
        reset_render_clock()

        try:
            while viewer.is_running() and (episodes == 0 or ep < episodes):
                action, _ = model.predict(obs, deterministic=True)

                # 在这个调用内部：
                # 1. ActionLatencyWrapper 处理动作延迟
                # 2. ActionNoiseWrapper 添加动作噪声
                # 3. base.step() 执行 frame_skip 个 mj_step()
                # 4. 每个 mj_step() 后调用 on_physics_substep()
                obs, reward, done, infos = venv.step(action)

                ep_ret += float(reward[0])

                # 这里不再调用 viewer.sync()。
                # Viewer 已经在 MuJoCo 子步回调中按 render_fps 同步。
                if done[0]:
                    idx = episode if episode is not None else ep
                    seedtxt = f" seed={s}" if s is not None else ""

                    print(
                        f"  ep {idx:3d}{seedtxt}  "
                        f"success={infos[0].get('is_success')}  "
                        f"dist={infos[0].get('dist', 0) * 1000:6.1f} mm  "
                        f"return={ep_ret:7.1f}"
                    )

                    ep += 1
                    ep_ret = 0.0

                    # DummyVecEnv 会在 done 时自动 reset 一次。
                    # 这里再次显式 reset 是为了应用指定 seed。
                    obs, s = seeded_reset(ep)

                    # MuJoCo reset 后 data.time 通常重新从 0 开始，
                    # 所以墙钟与仿真时间基准也必须一起重置。
                    reset_render_clock()

        finally:
            # 无论正常退出还是发生异常，都移除回调。
            base.set_physics_step_callback(None)

    venv.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize an SO-101 reach policy")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--episodes", type=int, default=0,
                   help="stop after N episodes (0 = run until you close the window)")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; makes episodes reproducible (episode k uses seed+k)")
    p.add_argument("--episode", type=int, default=None,
                   help="watch only this episode index, replayed (implies --seed 0 if unset)")
    p.add_argument("--randomize", action="store_true", help="run under domain randomization")
    p.add_argument("--which", type=str, default="best", choices=["best", "final"])
    p.add_argument("--video", type=str, default=None,
                   help="headless: render an mp4 instead of opening a window (set MUJOCO_GL=egl)")
    p.add_argument("--no-realtime", action="store_true", help="run the viewer as fast as possible")
    p.add_argument("--render-fps", type=float, default=60.0, help="interactive viewer refresh rate in simulation frames per second",)
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    if args.video:
        # reuse the tested eval recording path — no duplicated rollout logic
        from sim2real.eval import evaluate
        evaluate(run_dir, episodes=1, randomize=args.randomize, which=args.which, video=args.video)
    else:
        watch(run_dir, args.episodes, args.randomize, args.which,
              realtime=not args.no_realtime, seed=args.seed, episode=args.episode, render_fps=args.render_fps,)
        # The MuJoCo/GLFW viewer's native teardown can segfault during normal
        # Python shutdown. We're done and everything is flushed, so exit hard to
        # skip the buggy finalizers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
