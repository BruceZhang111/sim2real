# 可执行任务清单

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 已完成 · `[!]` 被阻塞

## 当前排序原则

自 2026-10-14 起，课程 tutorial assignment 是唯一 P0 主线。SO-101 sim-to-real 代码、训练产物和历史任务继续保留，但除非直接服务于 PPO+SimBa 实验或课程展示，否则不在 2026-11-05 前扩展。

## 当前执行入口：先统一评估，再补最小对照（2026-10-22）

本节覆盖下方早期优先级与服务器迁移待办。旧列表保留为历史，不把合并任务的部分完成误标为全部完成；最新证据见 STATUS.md 顶部。

### 成功率曲线导出（2026-10-26）

- [x] 编写21个截图run的TensorBoard eval/rollout success rate批量CSV入口；仅静态检查，按用户要求不执行。
- [ ] 用户同步scripts/export_success_rates.py至服务器，运行导出并传回CSV；核对配置与评估条件后再制图。

### 当前完成：扩大Critic初始化Fourier分析（2026-10-26）

- [x] 四种Critic各100初始化×300网格；本地CPU-only双进程，不连接服务器。
- [x] 统一function complexity/simplicity bias score术语，保留已确认的项目操作定义和论文差异。
- [x] 五张最终图及逐图详细中文REPORT.md，包含真实反预期结果、窗函数敏感性和幅值/功率口径差异。
- [x] 18项测试、400网格/800FFT全量复算、bootstrap汇总和源码快照校验、40组旧预览初始化权重一致性验证通过。
- [x] 扩大实验验证后按用户授权删除小规模预览、临时检查及旧图表导出，训练产物不变。
- [ ] 用户审阅最终报告，按需要安排输入域/分辨率/初始化单因素敏感性或训练后有效状态切片；不把空间高频等同动作抖动，不宣称高频已被证明有益。

### 历史阶段：本地Critic Fourier预览（2026-10-26）

- [x] 用户反馈已完成MLP256/1440、SimBa64/512、64/256、128/512训练，64/512最好；此项是反馈记录，未独立核对服务器原始评估。
- [x] 在用户指定外层Fourier目录构建独立入口；4种2D→1D Critic，10初始化×128网格，不连接仍在训练的服务器。
- [x] 保存原始浮点/完整复数FFT、矩形/Hann口径、40样本、80指标、汇总图/全seed图册、源码快照和中文说明。
- [x] 16项定向测试、两次同协议权重/网格完全复现、80组FFT复算和汇总/哈希验证通过。
- [ ] 用户审阅当前反预期结果（本协议下MLP的C更低）及边界敏感性，再决定128→256/300、输入域和初始化的分别对照；不选择性筛seed或改指标。
- [ ] 如需解释训练后行为，再另建有效状态切片入口；当前toy结果不是训练后V(s)，不能替代因果分析或多训练seed性能证据。

### 历史训练批次更新（用户反馈，2026-10-22）

- [x] 用户反馈A组seed1/2/3已跑完（仅训练完成反馈，服务器产物与性能尚未独立核验）。
- [x] 本地准备B-E seeds1/2/3串行脚本，addname为B1～E3；含dry-run、失败/中断停止、冻结配置及start-at选项。
- [ ] 用户同步脚本后执行B-E共12次2M训练（24M请求步），不重跑A。
- [ ] 核对五组15个run实际配置/最终步数与成对归一化，再统一final nominal/DR各200回合。
- 本批所有训练seeds统一为1/2/3；下方旧0/1/2计划只保留历史来源，不混入新seed集合。

### 本轮已核验

- [x] 确认外层与实际 Git 根目录、起始干净 main/2e503cc；只在 WSL 实际仓库更新三份收尾文档。
- [x] 只读审计 9 个 reach run 的 YAML、final/best 元数据、参数量、evaluations.npz 和 TensorBoard 原始指标；没有重跑训练/eval。
- [x] 核对 actor256/128/64、critic512/1024 与预算对照的配置混杂；确认 128/1024 的 2M/3M eval 数组前缀一致，不当作独立种子。
- [x] 核对新候选 best=849456、MLP revised_v2 best=1925000；确认 best 按 nominal reward、final stats 错配风险及 eval 无显式 seed/原始导出。
- [x] 记录用户截图中的 final 200-episode 结果和证据来源；该项仅表示结果转录/核对，非本轮重跑或多seed验证。
- [x] 基于本对话实际输出记录服务器已安装 uv/Python/torch CUDA、依赖检查/smoke 成功及直接 SSH Git/outputs 传输流程；不表示 uv.lock、调度策略或提速已验证。
- [x] 制定课程最小三组/五组矩阵、机制诊断与两阶段 Fourier 分析边界，并记录 D-021～D-023。

### 当前首要：评估可信度（实现已验证，服务器重评待执行）

- [x] 在本地给 eval 增加独立评估 seed、固定场景/随机流协议、逐 episode CSV/JSON/元数据导出，保存模型哈希、配置/代码版本、归一化路径及版本。
- [x] 给未来 best 保存同步 VecNormalize，严格按模型选择配对；历史缺失不覆盖、不伪造。课程主比较先使用 final+配套 final stats，检查 _find_model 的静默回退是否误报 which。
- [x] 对 seed/原始导出/模型统计量配对写针对性测试；按照 AGENTS 做风险匹配的验证，不自动启动长训练。
- [ ] 用户同步后，统一 deterministic final nominal/DR 各≥200回合、相同测试 seed/场景重评8组已有 revised_v2 reach；origin 不纳入本轮。记录成功、距离、episode长度和平滑性，保存 raw 数据。
- [ ] 重新核对服务器 freeze：原清单有 rich15 但无 tqdm；requirements.txt 重复 rich 且新增 <15 上限，pyproject.toml 缺 tqdm。保留原快照、统一依赖声明并补最终快照，确认软件/资产版本；旧seed0均不复用到新8env/KL0.33矩阵。锁文件与固定CUDA源仍待实现/验证。

实现说明与服务器执行命令见 EVALUATION.md；本地30项相关测试和短smoke已通过，未自动提交/推送或启动长训练。

### 新训练：最低三组，若讲分支扩展则五组

共同控制：revised_v2、相同 DR/VecNormalize/软件资产、2M、8 envs、512 rollout steps/env、batch1024、epochs18、lr3e-4、KL0.33、训练 seeds0/1/2；不是最优调参宣称。final 评估优先，不按测试集挑 best。

| 组 | 配置 | 新增训练数（seeds 0/1/2，全部新训） | 用途 |
| --- | --- | ---: | --- |
| A | MLP [256,256] | 3 | 公平小基线，旧8env/无KL不能替代 |
| B | MLP [1440,1440]，实测4,236,491参数 | 3 | 与C总容量差约0.37%，检验更大MLP是否真的退化 |
| C | SimBa64/512，blocks1/2 | 3 | 候选，多seed检验 |
| D | SimBa256/512，blocks1/2 | 3 | 与C只改actor |
| E | SimBa64/256，blocks1/2 | 3 | 与C只改critic，验证256→512收益 |

- [~] 五组独立YAML已准备于configs/architecture_8env；8env/KL0.33、seeds0/1/2、2M。已实例化核验参数与共同字段，独立8env短smoke及21项相关测试通过；服务器软件/资产版本尚待冻结。
- [ ] 完成 A/B/C 的3seed主比较；新增9次2M（18M环境步），旧seed0不复用。
- [ ] 如要在presentation主张 actor/critic不对称，完成D/E；五组共15次，全部新训，共30M请求环境步；旧8env/无KL及18env/KL0.33均不满足新配方。
- [ ] 可选：SimBa64/1024 三seed，定位critic512→1024的饱和/退化；先seed0资源预览，未齐全不作多seed结论。
- [ ] 统一导出每seed点、均值/训练seed方差、成功率/距离、学习曲线、参数量/墙钟成本；原始episode置信区间与训练seed方差分开。
- [ ] 若要控制分支参数分配，再扩展MLP独立net_arch，候选actor[172,172]/critic[2040,2040]（预估4,256,031）；当前接口不支持，此项非最低交付硬前置。
- [ ] 若大MLP未退化，如实报告；不为符合题目筛选seed。actor32、深度、其他优化器/学习率网格延后，课程时间优先证据闭环。

### Fourier 与机制分析（性能协议后进行）

- [ ] 核对论文/作者FFT代码与初始化：toy二维标量输出、300×300、[-100,100]^2、100初始化；先低分辨率/少量seed预览，明确项目初始化与论文初始化差别。
- [ ] 实现初始化simplicity-bias代理与训练后函数频谱两个独立入口；计算幅值/功率/DC/频率权重约定明确，常量/近常量和边界泄漏检查通过。
- [ ] 构造共同有效状态/目标平面，保持FK、TCP、target-TCP等关联约束；模型各加载配套冻结统计量。actor按确定性均值各维、critic按V(s)分别分析，不把噪声动作谱当函数谱。
- [ ] 生成函数灰度图+log频谱+径向谱/复杂度，并比较多个seed/状态切片；共同输入域与色标，裁剪/饱和/归一化影响单列。
- [ ] 在共同留出轨迹检查value误差/GAE、KL/clip、梯度与动作/输入敏感性；分开表示相关性、假设与已验证干预，不用EV或单张灰度图证明因果。
- [ ] 最小证据齐全后制作10分钟slides：问题与论文→受控性能/容量→actor/critic差异→Fourier代理→局限；团队10月30日登记、11月5日slides提交仍待确认，不标完成。

## 历史优先级：critic 1024 对照与服务器执行环境（2026-10-20）

- [x] 核验当前 YAML 为 actor 128 / critic 1024、blocks 1/2；配置可加载。此项仅表示配置核验，不代表训练完成。
- [ ] 整理 actor 128 / critic 512 的固定 seed、final nominal/DR 基线；用户反馈效果改善，内置名义结果已核验，DR 结论仍待验证。
- [ ] 固定 actor 128、blocks 1/2、seed、18 envs、KL 0.33、lr 3e-4、epochs 18 等条件，仅改变 critic 512→1024；核对资源后训练并保存独立标签/配置。性能、显存/内存和耗时均待验证。
- [ ] 收集服务器连接信息（地址/端口/用户名、VPN/跳板机、认证方式）、OS、CPU/GPU/驱动、磁盘、权限和调度规则；不在聊天中收集密码/私钥。
- [ ] 指导 WSL SSH 初次连接、确认主机指纹和密钥权限；未实际连接，不得标记完成。
- [ ] 清点本地代码提交、资产、依赖与可选模型产物；制定首次上传和 Git 后续同步流程。outputs/环境/权重受 ignore 影响，不能声称一次 clone 就完整复刻。
- [ ] 导出完整环境清单，按服务器系统及 CUDA 兼容性重建/迁移；验证同一代码 SHA、import、headless MuJoCo、小型 smoke 和设备选择，再安排正式训练。
- [ ] 按服务器规则选择 tmux/任务调度器，指导日志查看、断线保持、TensorBoard SSH 隧道和产物回传；所有路径与端口待确认，不擅自暴露公网端口。
- [ ] 记录 CPU 环境 step 和 GPU 更新耗时，确认服务器实际提速；不盲目改 DummyVecEnv/环境数量污染 width 对照。

## P0：确定课程方案与外部期限

- [x] 将课程作业方向写入 `PROJECT.md`、`STATUS.md`、`TASKS.md` 和 `DECISIONS.md`。
  - 主题分类：Deep learning architectures。
  - 暂定题目：**Scaling Deep Reinforcement Learning: Why Bigger MLPs Fail and How SimBa Helps**。
  - 核心论文：Lee et al., *SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning*，ICLR 2025。
- [ ] 在 2026-10-30 前确认并登记 4–6 人团队信息。
  - 待记录：成员、登记状态、论文讲解/代码/实验/制图/slides/演讲分工。
- [ ] 确认最终英文题目和一句话中心论点。
  - 验证：题目明确落在 deep learning architectures，而不只是泛泛介绍强化学习。
- [ ] 在 2026-11-05 前提交一套含团队信息的 slides 至 `bddl2017@163.com`。
  - 验证：最终文件已发送并留存发送记录；展示时长经彩排不超过 10 分钟。

## P1：形成可教学的 SimBa 研究叙事

- [~] 精读 SimBa，形成“问题—机制—证据—局限”笔记。
  - 问题：标准 MLP 在深度 RL 中随参数增加可能退化。
  - 机制：RSNorm、pre-LayerNorm residual feedforward block、post-LayerNorm 共同增强 simplicity bias。
  - 证据：MLP/SimBa 参数扩展、组件消融、SAC/PPO/METRA 跨算法结果。
  - 局限：PPO 证据来自 Craftax、1024 并行环境和 10 亿步，不能直接外推到 SO-101 reach。
- [x] 完成原论文函数复杂度、simplicity bias 测量、SimBa 架构和 Figure 4 组件分析的首轮精读。
  - 已核对：Fourier complexity 是频率加权平均，simplicity score 是其倒数在随机初始化分布上的期望近似；论文以 90,000 个二维网格点和 100 次随机初始化估计架构的初始化低频倾向。
  - 已核对：SimBa 主干为 RSNorm、线性嵌入、`d_h -> 4d_h -> d_h` 的 pre-LN residual FFN（ReLU）和 block stack 后的 final LayerNorm。
  - 已明确边界：低频只是函数复杂度的代理；Figure 4(a) 的未训练初始化 score 与 Figure 4(b) 的训练后 return 是相关性证据，不构成“simplicity bias 导致性能提升”的因果证明。
  - 已明确叙事：各组件多数已有先例，论文贡献重点是 RL scaling 问题、组合方案和跨任务实验，而非首次发明这些算子。
- [ ] 用通俗图示解释 simplicity bias 与 Fourier complexity score。
  - 验证：不依赖公式也能说明“高容量网络为何仍可能偏向简单函数”；公式只作为补充。
- [ ] 明确论文结论与本项目验证问题的边界。
  - 验证：slides 不把 SO-101 的小规模结果描述为对论文全部 benchmark 的复现。
- [ ] 建立参考文献与图片来源清单。
  - 验证：论文主张、图表和对比方法均可追溯，引用格式统一。

## P2：在 SO-101 reach 中实现 PPO+SimBa

- [x] 准备 actor width 单因素消融：`reach_forSimBa.yaml` actor 256→128，critic 512、blocks 1/2 和其他训练条件不变；不修改既有预算批次快照。
- [~] 比较 actor 128 / 256：`ppo_reach_20261020_104421-actor128-critic512` 最近 10 次内置名义成功率 0.98，旧 256/512、KL 0.33 为 0.925；已核验原始数组。固定 eval seed 的 final nominal/DR 与多 seed 仍待验证。
- [x] 增加训练预算自动化脚本：冻结同一配置，串行从头运行 200 万、300 万、500 万步，支持无副作用预览、失败/中断停止后续实验。
- [x] 核验预算批次三个 run 均达到目标：final 保存步数分别为 2,009,088 / 3,004,416 / 5,004,288；共享配置为 256/512、18 envs、KL 0.33。只读核验，不是本次重新训练。
- [ ] 对上述 200/300/500 万步 run 用配对归一化文件和固定 eval seed 汇总 nominal/DR，判断延长预算是否有效；尚无涌现或普遍泛化结论。
- [x] 核对 `TrainConfig.target_kl` 与 `PPO(target_kl=cfg.train.target_kl)` 的源码接线；默认 None，不与 clip_range 混淆。
- [x] 为训练入口增加 `--addname`，在新 run 目录名称末尾追加实验标签；参数只影响后续启动的训练。
- [x] 确认 PPO+SimBa 的最小接入边界。
  - 已验证：`self.policy` 与 `self.rollout_buffer` 是独立对象；buffer 不包含神经网络，`old_values`/`old_log_prob` 来自 rollout 时同一个 policy 实例的输出。
  - 已决策：保留 `RolloutBuffer`、`collect_rollouts()`、`PPO.train()` 和既有 `evaluate_actions()`，通过兼容 `MlpExtractor` 接口的 `SimBaMlpExtractor` 与 `ActorCriticPolicy._build_mlp_extractor()` 接入。
  - 当前实现：`SimBaActorCriticPolicy` 只覆写 `_build_mlp_extractor()`，PPO 算法与 buffer 均未改动。
- [x] 设计可配置的 policy architecture 接口。
  - 已实现：`policy_arch: mlp|simba`；SimBa actor/critic 的 hidden dim 与 block 数分别配置，4× expansion 与 ReLU 固定为论文结构。
  - 已验证：未知架构、SAC+SimBa、缺少 observation `VecNormalize` 均明确报错；默认 `mlp` 路径仍使用 SB3 `MlpPolicy` 与 `policy_hidden`。
- [x] 明确 RSNorm 与现有 `VecNormalize` 的关系并记录决策。
  - 已核对：当前 reach 使用 SB3 2.9.0 `VecNormalize`；observation 在每个 reset/step 逐维标准化，统计量跨 episode 和 8 个逻辑环境累计，训练时更新、评估时冻结，输出裁剪到 `[-10, 10]`。
  - 已核对：reward 使用折扣回报方差进行缩放且不减均值；action 是 `[-1, 1]` 固定坐标到 `0.1 rad` delta 的线性映射，后二者都不是 SimBa RSNorm。
  - 已决定：主比较统一保留环境级 `VecNormalize`，SimBa 网络内不重复加入 RSNorm；模型与 `vecnormalize.pkl` 仍是不可拆分的实验产物。
  - 消融边界：若比较网络内 RSNorm 或关闭 `VecNormalize`，必须作为单独实验并明确报告，不能与主架构变量混合。
  - 2026-10-20 修正：作者 RSNorm 实现是 agent wrapper，不等于 BatchNorm；不能只按网络内/外分类。当前 PPO 复用采样时归一化 obs 与 old_log_prob，不直接用新统计量重归一化旧 buffer；替代方案未实现（D-019）。
- [x] 实现 SimBa actor/critic 网络。
  - 已实现：actor `d_h=256`、`L=1`、bottleneck `1024`；critic `d_h=512`、`L=2`、bottleneck `2048`；均为 input Linear、pre-LN residual FFN、stack 后 final LayerNorm。
  - 已验证：前向 shape、动作裁剪、有限梯度、`evaluate_actions()`/`predict_values()`、一次 PPO update、独立 policy 与 PPO checkpoint save/load、跨进程加载和 ONNX Runtime parity。
  - 复现边界：本轮严格对齐论文主干拓扑，但保留 SB3 初始化与当前 PPO optimizer/learning rate；不是论文完整训练配方复现。
- [ ] 建立最小架构比较矩阵。
  - A：当前 PPO MLP `2×256`。
  - B：参数量更大的标准 MLP，用于验证“bigger MLP”是否退化。
  - C：参数量可比或更大的 PPO+SimBa。
  - 控制变量：同一 reach 环境、`revised_v2`、训练步数、seed、`n_envs`、评估条件和归一化口径。
- [~] 补全训练生命周期与资源记录。
  - 已完成：单环境 SO-101 reach 的 8-step rollout/update 单元级 smoke，无 NaN；论文起点在 reach `26→5` 接口下共 4,751,627 个可训练参数。
  - 已有：正式 SimBa 与三个预算 run 的配置、final 和 eval 产物；不再把首次正式训练列为未开始。
  - 待完成：全流程配对评估、wall-clock/显存或内存记录及公平对照。
  - 验证：训练无 NaN/OOM，checkpoint、VecNormalize、评估数组和参数统计完整。
- [ ] 运行正式多 seed 实验并生成课程图表。
  - 指标：success rate、final distance、episode length、sample efficiency、训练方差、参数量、wall-clock、动作差分与关节速度。
  - 最低输出：原始 CSV/NPZ、均值与置信区间、学习曲线、参数量—性能图和失败案例。
- [ ] 若时间允许，进行架构组件消融或宽度扫描。
  - 优先级：MLP width scaling > 去除 residual/pre-LN/post-LN 的最小消融 > 更大规模训练。

## P3：制作并交付 10 分钟 tutorial

- [ ] 完成逐页提纲与时间预算。
  - 建议结构：背景 1 分钟、问题与 simplicity bias 2 分钟、SimBa 架构 2 分钟、论文证据 2 分钟、PPO 实践 2 分钟、局限与结论 1 分钟。
- [ ] 完成第一版 slides。
  - 必含：团队信息、背景与动机、方法/模型、应用或实验、结论和引用。
- [ ] 将本项目实验图替换占位图，并加入公平比较说明。
- [ ] 至少完成两轮 10 分钟彩排和一次问答准备。
- [ ] 提交最终 slides 并准备第 8 周展示版本。

## 暂缓 A：原 reward 结论任务

- [x] 建立项目级 `AGENTS.md` 和 `docs/{PROJECT,STATUS,TASKS,DECISIONS}.md`。
  - 验证：文件位于 WSL Git 根目录且 `git status` 可见；业务代码 diff 不因本任务改变。
- [x] `outputs/ppo_reach_20261013_120248-revised_v2` 的 200 万步训练已自然结束。
  - 依赖：训练进程（已满足）。
  - 验证：`final_model.zip`、`vecnormalize.pkl`、`best_model/best_model.zip`、`eval/evaluations.npz`、200 万步 checkpoint 及其 VecNormalize 文件齐全；进程正常退出且无新 OOM 记录。
  - 结果：末次内置 eval 成功率 `0.90`；best-reward checkpoint 在 192.5 万步，内置 eval 成功率 `1.00`。仍需独立评估。
- [ ] 汇总 revised_v2 的 final 与 best 结果。
  - 依赖：当前训练完成。
  - 验证：记录名义动力学下的成功率、最终距离、episode length、mean reward；固定 seed 可复现。
- [ ] 在同一评估脚本和 seed 集合下比较 origin、revised、revised_v2。
  - 依赖：三个 run 及各自 `vecnormalize.pkl` 可用。
  - 验证：输出含 raw 数据的表格；清楚标注 best/final、reward 类型、seed、是否 randomize，不用平滑曲线末值代替原始数据。
- [ ] 检查 revised_v2 是否消除奖励漏洞。
  - 依赖：revised_v2 评估轨迹可采集。
  - 验证：成功 bonus 每 episode 最多一次；失败 episode 无 terminal bonus；报告 reward 分项、阈值内步数、最长连续保持、动作差分和关节速度。

## 暂缓 B：原实验可信度任务

- [ ] 为 `revised_v2` 增加单元测试。
  - 覆盖：前 9 个连续达标步不终止且 `is_success=False`；第 10 步终止并只加一次 bonus；离开阈值时计数清零；失败轨迹不获得 terminal bonus。
  - 验证：针对性 pytest 和完整 `make test` 通过。
- [ ] 给 origin/revised/revised_v2 运行相同的多 seed 实验。
  - 建议：至少 3 个 seed；正式数量受课程时间/算力预算影响，待确认。
  - 验证：报告均值、标准差/置信区间和每个 seed 的原始结果。
- [ ] 增加 reach 平滑性指标。
  - 候选：`mean(||a_t-a_{t-1}||²)`、关节速度 RMS、TCP 速度 RMS、阈值内抖动幅度。
  - 验证：eval 输出和 TensorBoard 可追溯，单位与定义写入报告。
- [ ] 改善 best checkpoint 选择标准。
  - 问题：SB3 `EvalCallback` 当前按 mean reward 保存 best。
  - 验证：设计按 success rate 优先、距离/平滑度作为次级指标的 callback，并确保兼容旧 run；实现前先记录决策。
- [ ] 修复 best 模型/归一化统计量配对，给独立 eval 增加固定 seed 与原始评估记录；不改写历史模型。
- [ ] 训练配置不存在时 fail-fast，并改进预算 dry-run 的“预览路径尚未创建”提示；当前两项尚未实现。
- [ ] 更新 README 中测试数量或取消易过时的硬编码数字。
  - 验证：README 与实际收集到的测试一致。

## 暂缓 C：原 SAC 对比与 PPO 调参

- [x] 建立 SAC reach 的 revised_v2 训练基线。
  - 配置：`configs/sac_reach.yaml`，与 PPO 使用相同环境、reward、seed、`n_envs=8` 和 200 万 environment timesteps；保留 SAC 的 replay buffer、learning starts、train frequency 与 tau。
  - 结果：`outputs/sac_reach_20261013_144914` 已生成 200 万步 checkpoint、best/final、评估数组和 VecNormalize；独立 nominal/DR、final/best 和平滑性评估仍未完成。
- [ ] 先建立不改变 reward 的 PPO 基线，然后一次只改变一组超参数。
  - 依赖：P0/P1 的 reward 结论稳定。
  - 关注：`approx_kl`、`clip_fraction`、entropy、policy std、success rate。
- [ ] 试验更保守的 PPO 更新。
  - 候选而非结论：`n_epochs≈8`、`learning_rate≈1e-4`、`clip_range≈0.15`、`ent_coef≈0.001`、`target_kl≈0.03`。
  - 验证：与当前 `n_epochs=18`、`lr=3e-4`、`clip=0.2` 做控制变量比较；不得同时改 reward。
- [ ] 评估 `n_envs=8 + DummyVecEnv` 的吞吐量、内存和训练质量。
  - 验证：记录 steps/s、RSS/WSL 内存峰值、训练时长和成功率；与必要的 4 env 基线比较。
- [ ] 明确 WSL 资源策略。
  - 待确认：是否通过 Windows 用户目录中的 `.wslconfig` 提高 memory/swap；修改前先核对主机可用内存并避免挤压 Windows。

## 暂缓 D：sim-to-real 验证

- [ ] 对候选 reach 模型执行 domain-randomized evaluation。
  - 依赖：选择稳定的名义动力学候选模型。
  - 验证：与 nominal 结果并列报告成功率、距离和平滑性退化。
- [ ] 导出候选 ONNX 并验证归一化已内嵌。
  - 验证：ONNX 与 SB3 在同一原始观测上的动作误差满足待确认阈值。
- [ ] 使用 MuJoCo stand-in 走完整部署闭环。
  - 验证：目标可达、关节限位和 `max_step_rad` 生效、控制频率匹配。
- [!] 完成 reach 实机验收。
  - 阻塞：真实 `center_ticks`、`joint_sign`、安全流程和验收指标待确认。
  - 解锁后验证：低步长短运行 → 多目标重复运行 → 记录实机/仿真 gap。
- [!] 完成 pick-and-place 实机闭环。
  - 阻塞：杯体位姿感知尚未接入。

## 暂缓 E：其他课程算法扩展

- [x] 选择一项与深度学习架构强相关、能在本项目中形成公平实验的方向。
  - 选择：SimBa + PPO，主题聚焦 MLP 参数扩展、simplicity bias 与可扩展 RL 架构。
- [ ] 仅在 SimBa 主线完成后评估 Transformer、Decision Transformer 或 model-based RL 是否需要作为补充相关工作。
- [ ] 先构建最小可比较接口，再实现算法。
  - 验证：新方法复用同一 env、观测、动作、seed 和 eval 指标；pipeline 变化与算法变化分开记录。
- [ ] 形成课程交付物。
  - 验证：论文综述、方法图、可复现实验配置、结果表/曲线、失败分析和现场演示材料齐全。

## Server migration checklist update (2026-10-20)

- [x] Capture local pip/conda/runtime inventories and correct direct dependency declarations.
- [x] Confirm with user that 3389 is the administrator-provided SSH port.
- [ ] Complete interactive SSH login and verify host fingerprint.
- [ ] Inspect server OS/GPU/driver/storage and scheduler policy.
- [ ] Confirm destination, synchronize code/assets/selected outputs, rebuild environment and verify imports/smoke.
- [ ] Record server environment and measure training throughput.

## VS Code and uv workflow (2026-10-20)

- [x] Record VS Code Remote-SSH GUI and uv workflow preference.
- [ ] Complete VS Code SSH login; inspect server resources.
- [ ] Configure project uv/PyTorch sources, lock dependencies against the recorded baseline, sync and validate the remote .venv.

## Local-to-server Git workflow (2026-10-20)

- [x] Verify existing GitHub origin and tracked XML/STL assets; document local-only AI development and Linux execution checkout.
- [ ] Connect via VS Code SSH; inspect resources, destination, GitHub access and scheduling policy.
- [ ] Prepare and validate uv metadata/lock locally against captured versions and server driver.
- [ ] Review/commit/push local changes, clone on server and compare commit SHA.
- [ ] Configure fast-forward-only pull and validate locked uv sync, GPU/MuJoCo and smoke.
- [ ] Arrange persistent/scheduled training and separate outputs transfer.

## Server inspection progress (2026-10-20)

- [x] User completed VS Code SSH login and supplied Linux/resource/tool inspection.
- [x] Record Ubuntu 22.04.2, driver 525.147.05, two busy 24 GiB GPUs; uv absent from PATH.
- [ ] Confirm exact GPU/CPU, project directory/quota, GPU allocation policy and GitHub access.
- [ ] Resolve a driver-compatible PyTorch/SB3 stack before local uv lock; CUDA 13 baseline cannot be copied unchanged.

## Git synchronization readiness (2026-10-20)

- [x] User verified exact GPU/CPU, system Python/Git and repository read access from server.
- [ ] User reviews, commits and pushes local migration preparation in WSL VS Code.
- [ ] Clone via server VS Code GUI into a confirmed directory; verify matching SHA.
- [ ] Prepare driver-compatible uv metadata/lock locally and sync server environment.

## Direct SSH Git synchronization (2026-10-20)

- [x] Record proxy-free Git-over-SSH choice and prepare local server remote.
- [ ] User creates/verifies bare receiver without overwriting existing paths.
- [ ] Review/commit local workflow edits and push main to server.
- [ ] Clone receiver into approved execution path, compare SHA and enable fast-forward-only pulls.
- [ ] Verify package download path and prepare compatible uv lock locally.

## Direct SSH synchronization verified (2026-10-20)

- [x] User verified receiver, matching local/server main SHA and clean execution checkout with local filesystem origin.
- [ ] Install user-scoped uv and verify package download access without the rejected proxy.
- [ ] Resolve compatible torch/SB3 versions; current SB3 2.9 requires torch>=2.8.
- [ ] Create/validate project Python3.11 lock locally, transfer and sync on server.

## uv bootstrap confirmed (2026-10-20)

- [x] User checksum-verified and installed uv 0.13.3 in server user directory.
- [ ] Install managed Python 3.11.15, check package access, prepare/test cu126 dependency candidate locally.
