# 项目当前状态

**最后更新：2026-10-13（Asia/Shanghai）**

## 本次初始化

- 已确认唯一工作仓库位于 WSL Ubuntu：`/home/brucezhang/Dev/Projects/SO101 Sim2Real/sim2real`；外层 `/home/brucezhang/Dev/Projects/SO101 Sim2Real` 是 VS Code 工作区，不是 Git 根目录。
- 已基于 README、现有 docs、配置、核心环境/训练代码、测试清单、Git 状态和本轮对话建立 `AGENTS.md`、`PROJECT.md`、`STATUS.md`、`TASKS.md`、`DECISIONS.md`。
- 本次只新增项目文档，没有修改业务代码、配置、训练产物或正在运行的训练进程。

## Git 与运行状态

- 分支：`main`，跟踪 `origin/main`。
- 文档初始化前已有未提交修改：`configs/reach.yaml`、`sim2real/config.py`、`sim2real/envs/so101_reach.py`、`sim2real/train.py`。这些修改实现并启用 `revised_v2`，不是本次文档初始化产生的。
- 本次检查开始时运行的命令为：
  `python -m sim2real.train --config configs/reach.yaml --reward-type revised_v2 --n-envs 8`
- 训练于 2026-10-13 12:27 自然完成 200 万步；最终目录为 `outputs/ppo_reach_20261013_120248-revised_v2`。final/best 模型、最终 checkpoint、评估数组和归一化文件均已核对存在，进程正常退出且没有新的 OOM 记录。

## 已完成内容

- SO-101 校准 MJCF、reach 与 pick-and-place 场景和 Gymnasium 环境。
- 基于关节位置目标的 delta/absolute 控制、控制频率匹配、关节限位、碰撞配置和正向运动学。
- PPO/SAC 训练入口、配置化超参数、TensorBoard、周期 checkpoint、best/final 模型、VecNormalize 保存与 warm start。
- 名义动力学评估、domain-randomized 评估、交互可视化、视频、ONNX 导出和 sim/实机部署入口。
- 动力学与命令链路随机化：质量/惯量、阻尼/摩擦/armature、执行器增益、重力、动作延迟和动作噪声。
- reach 奖励可通过 `reward_type` 选择：
  - `origin`：距离、近距离指数奖励、控制能量、关节速度和逐步成功奖励。
  - `revised`：在 origin 基础上加入相邻动作变化惩罚。
  - `revised_v2`：保留平滑项，将 near 权重由 0.4 降到 0.1；只在连续 10 步达标并完成 episode 时一次性给 20 分成功奖励。
- `info` 语义已拆分：`within_threshold` 表示当前步进入 2 cm 阈值，`is_success` 只表示连续保持后真正完成，另有 `success_hold_count`。
- 为保留 `n_envs=8` 又避免 SubprocVecEnv 的进程复制开销，训练环境已采用单进程 `DummyVecEnv` 顺序管理 8 个环境。
- 当前测试目录包含 30 个测试函数，覆盖配置、环境 API、奖励、DR、包装器、FK、抓取碰撞、课程重置和部署总线等。

## 当前正在进行

- `[P0]` 提取 final/best 的成功率、距离、episode length、动作平滑指标并与 origin/revised 同条件比较。
- `[P1]` 判断 `revised_v2` 是否同时保留 revised 的稳定性并恢复 origin 的成功率；随后进行多 seed 复验。

## 已有实验结果与分析

- run `ppo_reach_20261012_114650_nenvs8_origin` 与 `ppo_reach_20261012_164342_nenvs8-revised` 使用相同训练设置，主要区别为 reward 类型。
- 已有 eval 数据的原始末次成功率约为 origin `0.95`、revised `0.75`；最后 10 次评估均值约为 origin `0.99`、revised `0.87`。TensorBoard 平滑曲线的末端读数会不同，正式报告应以原始评估数组为准。
- revised 在可视化中更平滑，但失败 episode 的平均回报约 `121.2`，高于成功 episode 的约 `104.3`，且失败通常跑满 200 步。
- 对 revised final 的固定 seed 检查显示，失败轨迹曾进入成功阈值约 60.75 步，但最长连续保持仅 6–9 步；逐步 success bonus 仍累计了约 `303.75`，构成奖励漏洞。
- revised 失败 episode 的分项均值分析曾得到：distance `-6.08`、near `+46.97`、control `-3.67`、action-rate `-5.23`、velocity `-4.53`、success bonus `+303.75`。这直接推动了 `revised_v2` 的一次性终止奖励设计。
- revised 后期 PPO 指标曾出现约 `approx_kl=0.72`、`clip_fraction=0.79`、策略 `std=0.044`，提示更新过激和探索塌缩风险；尚未改 PPO 参数，以便先隔离 reward 变化的效果。
- `revised_v2` 的早期受控语义检查：前 9 个达标步 `within_threshold=True` 且 `is_success=False`，第 10 步才 `terminated=True`、`is_success=True` 并得到约 20 分的一次性终止奖励。该检查不等于完整训练效果验证。
- revised_v2 的训练内置名义评估共 80 个点、每点 20 episodes。200 万步末次评估：成功率 `0.90`、mean reward `17.725`、mean episode length `44.25`；按 mean reward 选出的 best 位于 192.5 万步，成功率 `1.00`、mean reward `19.683`、mean episode length `35.3`；最后 10 次评估成功率均值为 `0.945`。这些是单 seed、内置 eval 结果，尚不能替代独立的 final/best 对比。

## 当前阻塞问题

- 当前没有阻止仿真训练继续的代码级 blocker。
- `revised_v2` 已完成单 seed 训练，但独立 final/best 评估、动作平滑分项和多 seed 汇总尚未完成，不能据此确定最终方案。
- reach 实机定量验收阈值、真实机械臂标定值和多 seed 实验预算：**待确认**。
- pick-and-place 实机依赖外部物体位姿感知，当前尚未接入。

## 已知错误或风险

- WSL 曾因多进程环境复制触发 OOM 并中断训练；机器约 16 GB Windows 内存，WSL 当时可见约 7.6 GiB 内存和 2 GiB swap，且未发现 `.wslconfig`。这些数值来自此前排查，重启或配置变化后需重新确认。
- `n_envs=8` 现在由 `DummyVecEnv` 顺序执行，并不提供 8 个 Python 进程的并行加速；优点是内存显著降低。
- 当前 `EvalCallback` 的 best model 依据 mean reward 选择，而不是 success rate；奖励尺度改变后，best 不一定等于最高成功率模型。
- 单一 seed 不能证明 reward 优劣；origin/revised/revised_v2 的正式对比至少需要相同设置和多个 seed。
- `normalize_reward=True` 会改变训练时的奖励尺度；评估解释应区分环境原始回报与训练归一化后的信号。
- README 中“28 fast CPU tests”与当前扫描到的 30 个测试函数不一致，文档数字需要维护。

## 最近一次测试或仿真结果

- 最新训练是 revised_v2 的 200 万步 run；末次内置评估成功率 `0.90`，最后 10 次评估成功率均值 `0.945`。
- 最近的完整横向对比仍是 origin 与 revised 的 200 万步训练，结果见上方“已有实验结果与分析”；revised_v2 尚未完成同条件独立横向评估。
- 最近的受控功能检查是 `revised_v2` 连续保持/一次性奖励语义检查，结果符合预期。
- 本次文档初始化期间没有运行 pytest、smoke、额外 eval 或 viewer；训练在文档编辑期间自行正常结束。

## 下一步建议

1. 保留已核对齐全的 revised_v2 产物，记录目录改名后的最终路径，避免模型、配置和 VecNormalize 文件失配。
2. 在不改 PPO 参数的前提下，用与 origin/revised 相同的名义评估设置分析 revised_v2；同时评估 best 和 final，不只看 TensorBoard 平滑曲线。
3. 增加 reward 分项和动作平滑统计，验证失败轨迹不再从阈值反复进出中获利。
4. 用多个 seed 重复三种 reward 的训练/评估；先得出 reward 结论，再单独进行 PPO 学习率、epoch、clip/target-KL 调优。
5. 若 reward 已正确但 PPO 仍不稳，再试验更保守的更新参数；建议值属于待验证候选，不应直接写成结论。

## 当前工作依据的重要假设

- `outputs/` 中的 run 配置快照代表各次实际训练配置，比较时以快照而不是当前 `configs/reach.yaml` 为准。
- 当前 `revised_v2` 训练没有 warm start，因为命令行未传 `--init-from`。
- 用户希望保留 8 个环境带来的 rollout 多样性，当前接受用 DummyVecEnv 换取较低内存占用。
- 本文中的历史实验数字来自现有 run、TensorBoard/evaluations 分析和本轮对话；若要用于课程报告，应重新用脚本生成可复现表格。
