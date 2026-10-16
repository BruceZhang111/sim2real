# 项目当前状态

**最后更新：2026-10-16（Asia/Shanghai）**

## 当前优先级变更

- 当前首要目标已由继续推进完整 SO-101 sim-to-real 闭环，切换为完成 BDDL 课程 tutorial assignment。
- 课程主题归入 **Deep learning architectures**，暂定题目为 **Scaling Deep Reinforcement Learning: Why Bigger MLPs Fail and How SimBa Helps**。
- 核心材料为 Lee et al. 的 ICLR 2025 论文 *SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning*。成果形式是论文讲解、PPO 架构对比实验和不超过 10 分钟的教学型 slides。
- 当前仓库继续作为实验平台：使用既有 SO-101 reach、`revised_v2` 奖励、PPO 流水线和评估基础，比较标准 MLP、放大 MLP 与 SimBa。实机部署、pick-and-place 感知和其他 sim-to-real 扩展暂缓，但现有代码与产物必须保留。
- 课程硬约束：4–6 人一组；2026-10-30 前登记团队信息；2026-11-05 前向课程邮箱提交含团队信息的 slides；第 8 周展示；每组不超过 10 分钟。

## 本次初始化

- 已确认唯一工作仓库位于 WSL Ubuntu：`/home/brucezhang/Dev/Projects/SO101 Sim2Real/sim2real`；外层 `/home/brucezhang/Dev/Projects/SO101 Sim2Real` 是 VS Code 工作区，不是 Git 根目录。
- 已基于 README、现有 docs、配置、核心环境/训练代码、测试清单、Git 状态和本轮对话建立 `AGENTS.md`、`PROJECT.md`、`STATUS.md`、`TASKS.md`、`DECISIONS.md`。
- 本次只新增项目文档，没有修改业务代码、配置、训练产物或正在运行的训练进程。

## 本轮 PPO/SimBa 接口梳理（2026-10-16）

- 已基于当前 WSL 环境中安装的 Stable-Baselines3 2.9.0 源码，逐段核对 `PPO.train()`、`OnPolicyAlgorithm.collect_rollouts()`、`ActorCriticPolicy`、`MlpExtractor`、`DiagGaussianDistribution` 和 `RolloutBuffer` 的数据流。
- 已确认 `self.policy` 与 `self.rollout_buffer` 是 `PPO` 下的并列对象：神经网络输出由同一个 `ActorCriticPolicy` 实例计算，buffer 只保存 rollout 时的 `old_values`/`old_log_prob` 并计算 GAE advantages 与 returns，不包含神经网络。
- 已确认 PPO+SimBa 的最小接入边界：保留 `RolloutBuffer`、`collect_rollouts()`、`PPO.train()` 和既有 `evaluate_actions()`；在项目内实现兼容 `MlpExtractor` 接口的 SimBa actor/critic 主干，并用 `ActorCriticPolicy` 薄子类覆写 `_build_mlp_extractor()`。
- 当前 reach 接口已核对为 observation 26 维、连续 action 5 维；minibatch 中 `evaluate_actions()` 应继续返回 `values (B,1)`、`log_prob (B,)`、`entropy (B,)`。
- 本轮没有实现 SimBa 代码，没有运行 pytest、smoke、训练或性能实验。SimBa block 的精确结构、RSNorm 与 `VecNormalize` 的组合、optimizer 配置和保存/导出兼容性均为待验证。

## 本轮 SimBa 原论文精读与现有归一化核对（2026-10-16）

- 已精读并讨论原论文 Section 2.1、2.2、4 和 5.1 中与函数复杂度、simplicity bias score、RSNorm、residual feedforward block、pre/post LayerNorm 及组件分析直接相关的内容。
- 已明确论文的 simplicity score 是初始化阶段的代理指标：在固定二维输入网格上对未训练网络做 Fourier 分析，并对 100 次随机初始化取平均；它衡量低频倾向，不等同于普适的函数复杂度，也不能单独证明 simplicity bias 导致训练回报提升。
- 已明确 SimBa 的单个组件并非全新算子。RSNorm 的运行均值/方差递推属于既有在线统计方法；论文的主要贡献应表述为面向深度 RL 参数扩展的组件组合、实验验证和 simplicity-bias 解释，而不是首次提出 observation normalization、残差连接或 LayerNorm。
- 已只读核对当前 SB3 2.9.0 与项目接线：reach 原始 observation 为 `qpos/qvel/TCP/target/target-TCP/last_action`，训练时由环境外层 `VecNormalize` 对每一维维护跨 episode、跨 8 个逻辑环境的全局 running mean/variance，并在每个 reset/step 应用标准化和 `[-10, 10]` 裁剪；评估时统计量冻结并从训练环境同步或从 `vecnormalize.pkl` 加载。
- 当前 reward normalization 使用折扣回报 `G_t = gamma G_(t-1) + r_t` 的运行方差缩放单步奖励，`gamma=0.98`，不减 reward 均值；评估时关闭 reward normalization。当前 action 仅采用固定 `[-1, 1]` 坐标和 delta 模式下 `0.1 rad` 线性缩放，不属于 running-statistics normalization。
- 现有 observation `VecNormalize` 与论文 RSNorm 在数学目的上同类，但工程位置不同：前者是 policy 外部环境 wrapper，后者是 SimBa 网络入口组件。是否保留二者、移除其一或设置消融尚未决定；直接叠加会形成双重 observation normalization，正式实现前必须先固定公平比较方案。
- 本轮没有修改业务代码，没有运行 pytest、`make smoke`、训练或性能评估；上述结论来自原论文、当前项目代码、当前安装的 SB3 2.9.0 源码和配置的只读核对。

## Git 与运行状态

- 分支：`main`，跟踪 `origin/main`。
- 文档初始化前已有未提交修改：`configs/reach.yaml`、`sim2real/config.py`、`sim2real/envs/so101_reach.py`、`sim2real/train.py`。这些修改实现并启用 `revised_v2`，不是本次文档初始化产生的。
- 本次检查开始时运行的命令为：
  `python -m sim2real.train --config configs/reach.yaml --reward-type revised_v2 --n-envs 8`
- 训练于 2026-10-13 12:27 自然完成 200 万步；最终目录为 `outputs/ppo_reach_20261013_120248-revised_v2`。final/best 模型、最终 checkpoint、评估数组和归一化文件均已核对存在，进程正常退出且没有新的 OOM 记录。
- SAC revised_v2 的 200 万步训练也已于 2026-10-13 16:08 完成；目录为 `outputs/sac_reach_20261013_144914`，包含 10 万至 200 万步 checkpoint、best/final 模型、`evaluations.npz` 和 `vecnormalize.pkl`。截至 2026-10-14 检查时没有训练进程。该 run 可保留为历史算法对比，但不是当前课程作业的首要实验。

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

- `[P0]` 把课程要求、SimBa 论文主线和本仓库可执行的 PPO 实验整合为课程方案。
- `[P0]` 明确 10 分钟 tutorial 的中心论点：参数更多并不自动带来更好的深度 RL；带有 simplicity bias 的架构才使规模化有效。
- `[P1]` 设计标准 MLP、放大 MLP、PPO+SimBa 的控制变量实验，并确定最小可完成的参数规模与 seed 预算。
- `[P1]` SB3 接入路径已缩小为自定义 `SimBaMlpExtractor` + `ActorCriticPolicy` 薄子类；论文 block 结构和当前归一化接线已核对，RSNorm/`VecNormalize` 的主比较与消融方案仍待决定，网络实现和训练效果仍待验证。
- 原有 revised_v2 final/best、多 seed、DR、ONNX 和实机任务转入保留队列；只保留与课程实验直接相关的评估工作为近期任务。

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
- 课程团队成员、登记状态和组内分工尚未记录；2026-10-30 是最近的外部期限。
- PPO+SimBa 尚未实现；当前 `sim2real/train.py` 只通过 SB3 `MlpPolicy` 和 `policy_hidden` 构建标准 MLP。
- 课程实验算力和 seed 预算尚未确定，因此不能直接照搬论文 1024 并行环境、10 亿 environment steps 的 Craftax PPO 设置。
- 网络内 RSNorm 与环境级 `VecNormalize` 已确认属于同类 observation running-statistics 标准化但位于不同层级；主实验是否保留现有 wrapper、如何避免双重归一化及如何设置消融尚未决定，若处理不当会把归一化差异误当成架构收益。
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
- 2026-10-16 最近两轮仅完成 SB3 接口核对、SimBa 原论文精读和现有归一化实现的只读检查，没有运行 pytest、smoke、额外 eval、viewer 或训练；因此没有新增模型性能结论。

## 下一步建议

1. 在 2026-10-30 前确认 4–6 人团队、登记状态和每人的讲解/实验分工。
2. 把 10 分钟内容压缩为问题、机制、证据、PPO 实践、结论五个部分，先形成逐页提纲和时间预算。
3. 先确定 RSNorm 与现有 `VecNormalize` 的公平比较矩阵，明确主实验是否允许双重 observation normalization；记录决策后再实现可配置的 `mlp`/`simba` policy architecture。
4. 先跑最小实验矩阵：当前 `2×256` MLP、参数量更大的 MLP、参数量可比的 SimBa；固定环境、奖励、训练预算和 seed。
5. 若最小实验显示清晰趋势，再增加宽度扫描、组件消融和多 seed；不要在课程期限前同时扩展奖励、PPO 超参数和 sim-to-real 部署。
6. 在 2026-11-05 前完成 slides、图表、引用、团队信息和 10 分钟彩排版本。

## 当前工作依据的重要假设

- `outputs/` 中的 run 配置快照代表各次实际训练配置，比较时以快照而不是当前 `configs/reach.yaml` 为准。
- 当前 `revised_v2` 训练没有 warm start，因为命令行未传 `--init-from`。
- 用户希望保留 8 个环境带来的 rollout 多样性，当前接受用 DummyVecEnv 换取较低内存占用。
- 课程作业允许选择 **Deep learning architectures**，并允许用 PPO+SimBa 的本项目实践作为教学案例；最终题目仍为暂定，后续可在不改变研究主线的前提下精简措辞。
- 本文中的历史实验数字来自现有 run、TensorBoard/evaluations 分析和本轮对话；若要用于课程报告，应重新用脚本生成可复现表格。
