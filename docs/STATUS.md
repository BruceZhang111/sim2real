# 项目当前状态

**最后更新：2026-10-20（Asia/Shanghai）**

## 本轮收尾：已核验现状与下一轮交接（2026-10-20）

本节是当前状态；后文按日期保留历史记录，不能把早期“尚未训练”的描述理解为当前状态。

### 目录、Git 与操作范围

- 已确认外层工作目录为 `/home/brucezhang/Dev/Projects/SO101 Sim2Real`；实际 Git 仓库及本文档位置为其下的 `sim2real/`。不得在外层另建一套 docs。
- 收尾开始时 `main` 工作树干净，跟踪 `origin/main`，最新本地提交为 `b4f6376`（`manySimBaStuff`）；本轮收尾不创建 commit、不 fetch/push、不操作服务器。
- 当前收尾仅修改 `docs/STATUS.md`、`docs/TASKS.md`、`docs/DECISIONS.md`。项目目标/范围未变化，服务器执行流程尚待落地，因此不修改 `PROJECT.md` 或新增 `AGENTS.md` 长期操作规则。
- 本次进程检查未发现 Python 训练或预算脚本进程；这是检查时刻的快照，不代表持续监控。

### 已有代码及当前配置

- 已核对 `train.py` 的 `--addname` 命名功能、`TrainConfig.target_kl` 到 `PPO(target_kl=...)` 的传递，以及串行预算脚本和对应测试均存在于当前提交中；KL 设置为 `None` 时不启用提前停止。
- 当前 `configs/reach_forSimBa.yaml` 已由用户改为 actor width **128** / critic width **1024**，blocks **1/2**，inverted bottleneck 为 **512/4096**。本次只验证配置加载，不再次修改 YAML；critic 1024 的训练效果**待验证**。
- 其余当前设置：`n_envs=18`、`n_steps=512`、`batch_size=1024`、`n_epochs=18`、`learning_rate=3e-4`、`target_kl=0.33`、`seed=0`，训练 DR 开启；YAML 仍写 `policy_arch: mlp`，直接训练必须显式传 `--policy-arch simba`。预算脚本默认已显式启用 SimBa。
- YAML 中旧的“shrink actor only / keep critic width”注释对应此前 128/512 消融，现已与用户后续设定的 critic 1024 不一致；后续整理配置时应更正，本次文档收尾不代改配置。

### 训练预算与 width 实验的证据

- 批次 `outputs/timestep_sweep_20261019_222032_a097e2ee/` 的共享快照仍为 actor 256 / critic 512、blocks 1/2、18 envs、KL 0.33。三个实验是同 seed 独立从头训练，不是续训。
- 已只读检查 `final_model.zip` 中保存的 `num_timesteps` 和评估数组；三组均达到预算，PPO 完成 rollout 导致略超步数。仅证明预算与产物存在，不代表性能验证完成或恢复训练严格等价。

| 该批次中的 run | 请求总步数 | final 保存步数 | 最后内置 eval 步数 |
| --- | ---: | ---: | ---: |
| `ppo_reach-simba-ts2000000` | 2,000,000 | 2,009,088 | 1,998,720 |
| `ppo_reach-simba-ts3000000` | 3,000,000 | 3,004,416 | 2,998,080 |
| `ppo_reach-simba-ts5000000` | 5,000,000 | 5,004,288 | 4,996,800 |

- actor 128 / critic 512 的 run 为 `outputs/ppo_reach_20261020_104421-actor128-critic512`，final 保存步数为 2,009,088。用户反馈 width 调整后效果变好；本次核验其最后 10 次**内置名义评估**：成功率均值 0.98、回报 18.631、长度 37.34；旧 256/512、KL 0.33 run 对应为 0.925、16.633、59.45。该比较支持本次单 seed 的名义性能改善，不证明 DR 改善或普遍优于 MLP。
- 用户此前截图报告：256/512、KL 0.33 的 best 模型 nominal 46/50、DR 19/50；MLP best nominal 45/50、DR 42/50。来源为用户终端截图，本次未重新执行；固定评估 seed、多 seed 和 best 归一化配对仍**待验证**。原始 MLP 的 `n_epochs` 确认为 **18**；其 8 envs 与这些 SimBa run 的 18 envs 也是混杂因素。
- 200/300/500 万预算实验的固定条件 final nominal/DR 横向汇总，以及“延长预算是否改善鲁棒性”仍**待验证**，不能宣称出现 grokking 或涌现。

### 已完成验证与未解决风险

- 本次重新运行 `tests/test_timestep_experiments.py` 的 **8 项 unittest：通过**；子进程均为 mock，没有启动 PPO 或仿真。覆盖串行顺序、冻结配置、dry-run 无写入/训练、失败停止和中断停止。
- 已检查 `git status` 和三份文档的 `git diff`，`git diff --check` 通过；收尾后未提交差异仅为三份指定文档，业务代码、YAML、PROJECT 和 AGENTS 不变。
- 对 `train.py`、`config.py`、预算脚本进行内存编译语法检查通过；当前 YAML 加载及 128/1024、1/2 参数断言通过。完整 pytest、smoke、模型前向与 1024-width 的训练/导出本次**未运行**。
- 2026-10-16 的 13 项 SimBa 定向测试与完整测试中的既有成功条件用例失败仅作为历史结果保留，不能描述为本次已重新通过。
- `eval.py` 默认 best，但读取根目录 final 的 `vecnormalize.pkl`；best 与统计量可能不匹配。评估脚本没有显式 eval seed；回调按干净环境 mean reward 选 best，而非 DR 鲁棒性。上述实现改进均尚未完成。
- `train.py` 对不存在的 YAML 会回退默认 Config；dry-run 打印的 shared_config 路径并不创建文件，直接复制该预览子命令可能误跑默认 MLP。修复 fail-fast 与预览提示**待实施**。
- RSNorm 与 VecNormalize 数学目的同类，不等于实现行为完全一致；作者实现是 agent wrapper，关键在统计更新/使用时机和 buffer 是否保存原始 obs。当前不叠加第二套归一化，不在预算或 width 实验中混入归一化改动；详见 D-019。

### 下一轮优先事项

1. 先核对 actor 128 / critic 512 的评估证据与预算结果；固定 actor 128、blocks 1/2 及其他训练条件，验证 critic **512→1024** 的单因素对照。当前配置已是 1024，但未发现对应训练结果，需先做轻量资源检查再安排运行，不自动启动。
2. 指导用户从本地 WSL Ubuntu 连接服务器：先收集主机地址、端口、用户名、VPN/跳板机、认证方式、OS、CPU/GPU/驱动、磁盘、权限及调度器信息。服务器路径和资源均**待确认**；不要求在聊天中提供密码或私钥。
3. 服务器迁移目标是“本地 WSL 编辑与 Git 版本管理 → 同步同一提交到服务器 → 服务器执行”。当前尚未连接、上传、配置 SSH 或同步。代码、资产、环境和模型产物要分别处理；Git 不包含被忽略的 outputs、权重及环境。
4. 按本地依赖清单在服务器重建并验证环境，不直接假定复制 conda 文件夹即可运行；若用户确需整包环境迁移，先核验系统/架构与驱动兼容。保留代码 SHA、实际配置、依赖版本和成对模型/VecNormalize，先 import/小型 smoke，再长训练；后台任务或调度方式依服务器规则选择。
5. 本地环境只读基线：Python 3.11.15、torch 2.14.0、MuJoCo 3.13.0、Gymnasium 1.3.0、SB3 2.9.0、NumPy 2.4.6。这不是完整依赖锁文件；CUDA 兼容性、完整导出/锁定和服务器实测均**待完成**。DummyVecEnv 环境 step 仍在 CPU，服务器 GPU 是否提速必须实测，不保证换卡即加速。

## 历史：SimBa actor width 消融准备（2026-10-20）

- `configs/reach_forSimBa.yaml` 的 actor hidden dim 从 256 改为 128，critic 保持 512，block 数保持 actor 1 / critic 2；仅改变 actor width，环境、奖励、seed、KL、学习率和 epoch 等均不变。actor 的 4 倍 bottleneck 随之变为 512，critic 仍为 2048。
- 该 YAML 的 `policy_arch` 仍为 `mlp`，未擅自修改；直接调用训练入口需显式传 `--policy-arch simba`，预算脚本默认已显式启用 SimBa。
- 已核对预算批次 `outputs/timestep_sweep_20261019_222032_a097e2ee/shared_config.yaml` 冻结为 actor 256 / critic 512；本轮不修改该快照或任何历史输出，因此不改变这批预算实验的条件。
- 此记录对应当时仅 actor 256→128 的配置准备；后续 run 和用户进一步设置 critic 1024 的当前状态见上方收尾节。论文 Figure 13 来自 SAC，不能直接保证 PPO 同样受益。

## 训练预算自动化实验（2026-10-19）

- 新增 `scripts/run_timestep_experiments.py`：串行从头运行 200 万、300 万、500 万步 PPO 实验；默认显式启用 SimBa，其余参数来自同一份启动时冻结的配置，seed、环境、奖励、DR、采样和优化设置不变。
- 使用 `python scripts/run_timestep_experiments.py --dry-run` 预览；去掉 `--dry-run` 才启动。默认读取 `configs/reach_forSimBa.yaml`，可用 `--config` 指定其他配置；拒绝非空 `init_from`，失败或 Ctrl+C 后不再启动后续实验。
- 每批在配置的输出根目录下创建独立 `timestep_sweep_<时间>_<随机标识>/`，保留共享配置、命令清单和三个独立 run。PPO 为完成 rollout 可能略超预算；配置被冻结，但运行期间仍应保持源代码不变。
- 本轮仅进行轻量脚本验证，不启动正式训练或仿真，不修改训练入口、已有 YAML 或历史输出。
- 验证：8 项不启动真实子进程的单元测试通过（含源 YAML 在两次实验间被修改时仍复用同一快照）；真实配置 `--dry-run` 与 `git diff --check` 通过。完整训练与性能结论仍待用户执行。

## 训练目录命名补充（2026-10-19）

- `train.py` 新增 `--addname`：在自动生成或 `--run-name` 指定的目录名称后追加 `-<后缀>`，例如 `--addname lr=0.0001`。此参数仅作命名标签，不修改学习率。
- 已通过轻量语法、命名和 CLI 透传检查。当前训练进程继续运行，未修改已有输出目录；本轮没有启动模型、仿真或完整测试。

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
- 截至该接口梳理阶段尚未实现 SimBa；随后完成的代码与验证见下方“本轮 SimBa 最小实现”。

## 本轮 SimBa 原论文精读与现有归一化核对（2026-10-16）

- 已精读并讨论原论文 Section 2.1、2.2、4 和 5.1 中与函数复杂度、simplicity bias score、RSNorm、residual feedforward block、pre/post LayerNorm 及组件分析直接相关的内容。
- 已明确论文的 simplicity score 是初始化阶段的代理指标：在固定二维输入网格上对未训练网络做 Fourier 分析，并对 100 次随机初始化取平均；它衡量低频倾向，不等同于普适的函数复杂度，也不能单独证明 simplicity bias 导致训练回报提升。
- 已明确 SimBa 的单个组件并非全新算子。RSNorm 的运行均值/方差递推属于既有在线统计方法；论文的主要贡献应表述为面向深度 RL 参数扩展的组件组合、实验验证和 simplicity-bias 解释，而不是首次提出 observation normalization、残差连接或 LayerNorm。
- 已只读核对当前 SB3 2.9.0 与项目接线：reach 原始 observation 为 `qpos/qvel/TCP/target/target-TCP/last_action`，训练时由环境外层 `VecNormalize` 对每一维维护跨 episode、跨 8 个逻辑环境的全局 running mean/variance，并在每个 reset/step 应用标准化和 `[-10, 10]` 裁剪；评估时统计量冻结并从训练环境同步或从 `vecnormalize.pkl` 加载。
- 当前 reward normalization 使用折扣回报 `G_t = gamma G_(t-1) + r_t` 的运行方差缩放单步奖励，`gamma=0.98`，不减 reward 均值；评估时关闭 reward normalization。当前 action 仅采用固定 `[-1, 1]` 坐标和 delta 模式下 `0.1 rad` 线性缩放，不属于 running-statistics normalization。
- 现有 observation `VecNormalize` 与论文 RSNorm 在数学目的上同类。2026-10-20 补充核对：作者实现是 agent wrapper，区别不应简化为“网络内/外”，还包括统计量更新时间和 replay/rollout buffer 的存储与使用方式。主比较继续保留 `VecNormalize`，不叠加第二次归一化；另做消融需保证 PPO 概率比率的输入一致性（D-016、D-019）。
- 截至原论文精读阶段尚未修改业务代码或运行测试；随后实现与验证结果见下一节。

## 本轮 SimBa 最小实现（2026-10-16）

- 已新增 `sim2real/policies/simba.py`，包含 `SimBaBlock`、兼容 SB3 `MlpExtractor` 契约的 `SimBaMlpExtractor`，以及只覆写 `_build_mlp_extractor()` 的 `SimBaActorCriticPolicy`；未修改 `PPO.train()`、rollout buffer、GAE、动作分布或 PPO loss。
- 网络严格按论文主干拓扑实现：已归一化 observation 先经无激活的 Linear；每个 block 为 `LayerNorm → Linear(d_h,4d_h) → ReLU → Linear(4d_h,d_h) → residual add`；所有 block 后再做 final LayerNorm。actor 使用 `d_h=256, L=1`，critic 使用 `d_h=512, L=2`。
- `TrainConfig`、`configs/reach.yaml` 和 CLI 已增加 `policy_arch: mlp|simba` 及 SimBa hidden/block 参数。默认仍为 `mlp`；使用 `--policy-arch simba` 启用。SimBa 当前只支持 PPO，并要求实际传入启用 observation normalization 的 `VecNormalize`。
- 网络内没有 RSNorm 或其他运行统计模块；现有 `env_factory` 的 `VecNormalize` 是唯一 observation running-statistics normalization，评估、保存与 ONNX 仍沿用原有统计量生命周期。
- `tests/test_simba_policy.py` 的 13 项定向测试通过，覆盖配置加载、精确层次/维度、shape、动作范围、梯度、SB3 policy 契约、Pendulum 与真实 SO-101 reach 各一次 8-step PPO update、默认 MLP 路径、错误配置、独立 policy/PPO save-load、跨进程加载及 ONNX Runtime parity。
- reach 的 26 维 observation、5 维 action 配置下，该论文起点共有 4,751,627 个可训练参数。当前 `2×256` MLP 约 146,955 个参数，因此第一轮论文配置不是参数量公平匹配，正式实验还需加入 matched-size 对照。
- 本轮只保证论文 SimBa 主干拓扑；初始化、单一 SB3 optimizer 和学习率仍沿用项目基线，没有同时切换到论文附录的 AdamW/`1e-4` 配方。尚未运行正式 SO-101 训练，因此没有新增性能结论。

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
- 当前测试还新增 SimBa 结构、梯度、SB3 接线、保存/加载与 ONNX parity 覆盖。

## 当前正在进行

- `[P0]` 把课程要求、SimBa 论文主线和本仓库可执行的 PPO 实验整合为课程方案。
- `[P0]` 明确 10 分钟 tutorial 的中心论点：参数更多并不自动带来更好的深度 RL；带有 simplicity bias 的架构才使规模化有效。
- `[P1]` 设计标准 MLP、放大 MLP、PPO+SimBa 的控制变量实验，并确定最小可完成的参数规模与 seed 预算。
- `[P1]` 自定义 SimBa 接线、预算实验产物与 actor 128 / critic 512 单 seed 名义改善已核验；下一步是 critic 1024 对照、预算 nominal/DR 汇总和服务器迁移指导，多 seed 与泛化结论仍待验证。
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
- PPO+SimBa 已有正式训练和初步比较；尚缺固定 seed 的 paired final nominal/DR 汇总、多 seed、参数量匹配对照，以及 critic 1024 的训练证据。
- 课程实验算力和 seed 预算尚未确定，因此不能直接照搬论文 1024 并行环境、10 亿 environment steps 的 Craftax PPO 设置。
- 主实验的归一化口径已确定为只使用环境级 `VecNormalize`；若后续做 RSNorm/归一化位置消融，必须单独配置和报告。
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
- 论文 PPO+SimBa 起点约 475 万参数，远大于当前约 14.7 万参数的 `2×256` MLP；不能把两者的直接结果解释成参数量公平的纯架构比较。
- 当前保留 SB3 初始化、Adam 和项目学习率，尚未复现论文附录的完整 PPO optimizer 配方；任何报告必须区分“拓扑复现”与“训练 recipe 复现”。

## 最近一次测试或仿真结果

- 截至 2026-10-20，最新核验的 width run 为 actor 128 / critic 512 的 200 万步实验，最近 10 次内置名义成功率 0.98；原 revised_v2 MLP 的结果作为历史基线保留。详见本页收尾节。
- 最近的完整横向对比仍是 origin 与 revised 的 200 万步训练，结果见上方“已有实验结果与分析”；revised_v2 尚未完成同条件独立横向评估。
- 最近的受控功能检查是 `revised_v2` 连续保持/一次性奖励语义检查，结果符合预期。
- 2026-10-16 SimBa 定向测试 13 项通过；完整 pytest 除既有 `test_success_when_target_at_tcp` 与 `success_hold_steps=10` 语义不一致而失败外，其余测试通过。已完成单环境 SO-101 reach 8-step PPO update，但没有新增模型性能结论。

## 下一步建议

1. 在 2026-10-30 前确认 4–6 人团队、登记状态和每人的讲解/实验分工。
2. 把 10 分钟内容压缩为问题、机制、证据、PPO 实践、结论五个部分，先形成逐页提纲和时间预算。
3. 先完成本页收尾节中的 critic 1024 对照准备和 WSL→服务器执行环境迁移指导；不把已完成的正式训练误写成尚待首次短训练。
4. 再跑最小实验矩阵：当前 `2×256` MLP、参数量更大的 MLP、论文起点 SimBa，并补充参数量匹配对照；固定环境、奖励、训练预算和 seed。
5. 若最小实验显示清晰趋势，再增加宽度扫描、组件消融和多 seed；不要在课程期限前同时扩展奖励、PPO 超参数和 sim-to-real 部署。
6. 在 2026-11-05 前完成 slides、图表、引用、团队信息和 10 分钟彩排版本。

## 当前工作依据的重要假设

- `outputs/` 中的 run 配置快照代表各次实际训练配置，比较时以快照而不是当前 `configs/reach.yaml` 为准。
- 当前 `revised_v2` 训练没有 warm start，因为命令行未传 `--init-from`。
- 用户希望保留 8 个环境带来的 rollout 多样性，当前接受用 DummyVecEnv 换取较低内存占用。
- 课程作业允许选择 **Deep learning architectures**，并允许用 PPO+SimBa 的本项目实践作为教学案例；最终题目仍为暂定，后续可在不改变研究主线的前提下精简措辞。
- 本文中的历史实验数字来自现有 run、TensorBoard/evaluations 分析和本轮对话；若要用于课程报告，应重新用脚本生成可复现表格。

## Server migration preparation (2026-10-20)

Captured the actual local environment in docs/environment/wsl-20261020/; pip check passed. Corrected the requirements hardware note and declared torch directly in requirements.txt and pyproject.toml. Administrator-confirmed SSH port: 3389. Server login, OS/GPU/driver, destination, upload and installation remain pending. See docs/MIGRATION.md. No training or remote mutation performed.

## VS Code and uv workflow (2026-10-20)

User selected VS Code Remote-SSH GUI for connection and uv for the server Python environment. Migration guide updated accordingly; server login, uv installation, lockfile and remote environment creation are pending. Local conda and captured inventory remain unchanged.

## Local Git development / Linux execution (2026-10-20)

User confirmed Linux server and local-only AI source editing. Verified existing origin https://github.com/BruceZhang111/sim2real.git and 18 tracked asset files. Selected flow: local WSL edit/check/commit/push -> server clone or fast-forward pull -> locked uv environment sync -> training -> artifact/log return. Rewrote docs/MIGRATION.md and recorded workspace boundaries in AGENTS.md. Server resources, GitHub access, authentication, destination, uv install/lock/sync and training remain pending. Local migration changes are not yet committed/pushed. No server source edits or remote execution performed.

## Authenticated server inspection supplied by user (2026-10-20)

User logged in through VS Code Remote-SSH as zyg@linux; home /home/zyg; Ubuntu 22.04.2 x86_64. NVIDIA driver 525.147.05, two GPUs with 24564 MiB each (exact models truncated), utilization 90%/99% and memory used 17399/19388 MiB at snapshot. RAM available 210 GiB; shared /home free 6.8T, quota unknown. Git/Python3 present; uv/sbatch not found in PATH. No scheduler absence or GPU allocation may be inferred. Local torch 2.14.0+cu130 is not a compatible default for this driver. Keep local environment unchanged; do not upgrade shared driver. Record in docs/environment/server-20261020/inspection.json. Next: exact GPU/CPU models, GPU allocation policy, project path/quota, GitHub access; then choose a compatible PyTorch build and validate dependency constraints before locking. Clone/push/uv installation/training remain pending.

## Server hardware and GitHub access confirmed by user (2026-10-20)

Both GPUs are RTX 4090 (24564 MiB each), currently 99%/100% utilized in the latest supplied snapshot. CPU is dual Xeon Gold 6253CL, 36 physical cores / 72 logical CPUs. System Python 3.10.12; Git 2.34.1. User's server-side git ls-remote succeeded, returning origin HEAD 4894fb73b3821369fd23aab3427fa88142af5dce. This verifies repository read access at that time, not server push access or package-index access. Choose Python 3.11 for the project via uv; preserve system Python. Next: local review/commit/push of migration preparation, remote GUI clone into a confirmed project directory, then locally prepared driver-compatible uv lock and remote validation. uv.lock does not exist yet. GPU allocation and user quota remain unknown; no training started.
