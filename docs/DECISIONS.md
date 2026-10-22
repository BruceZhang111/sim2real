# 技术决策记录

只记录已在代码或当前工作方式中实际采用的决策。编号一经引用不复用；新决策追加在文件末尾。

## D-001：以 WSL Git 仓库为唯一项目工作区

- 日期：2026-10-13
- 决策：所有代码和项目文档直接维护在 `/home/brucezhang/Dev/Projects/SO101 Sim2Real/sim2real`，不在 Windows 创建同步副本。
- 原因：VS Code 当前连接 WSL，Linux 路径才是实际 Git 仓库和运行环境；双副本会造成误改、版本分叉和运行文件不一致。
- 放弃方案：在 `C:\Users\...` 下复制仓库后修改；通过不明确的 Windows 映射路径编辑。
- 影响：每轮任务开始必须验证 `pwd`、Git 根目录和状态；若不能访问 WSL，应停止而不是创建替代文件。
- 状态：已采用。

## D-002：以动作/观测可实机复现为 sim-to-real 接口原则

- 日期：既有设计，准确日期待确认
- 决策：策略输出归一化的关节位置目标；reach 观测只包含关节角/速度、由同一 MJCF 做 FK 得到的 TCP、任务目标及可选 last action。
- 原因：STS3215 以位置目标驱动，上述观测均可从伺服反馈、任务输入和共享 FK 得到。
- 放弃方案：仿真专属状态、不可在实机获得的 privileged observation；直接输出与实机接口不一致的力矩动作。
- 影响：新增观测必须说明实机来源；pick-and-place 的杯体位姿明确需要外部感知。
- 状态：已采用。

## D-003：以单份 YAML 定义并快照每次实验

- 日期：既有设计，准确日期待确认
- 决策：所有可调参数由 dataclass 配置和 `configs/*.yaml` 管理，训练启动时将完整配置写入 run 目录。
- 原因：减少隐藏参数，使模型、奖励、DR、训练和部署设置可追溯；未知键可尽早失败。
- 放弃方案：散落的模块常量、只依赖命令历史、用多份几乎相同的 Python 文件表示实验。
- 影响：评估旧模型应读取 run 自带 `config.yaml`；命令行 override 只影响本次 config，必须进入快照。
- 状态：已采用。

## D-004：保存 VecNormalize 并在评估/部署中保持一致

- 日期：既有设计，准确日期待确认
- 决策：训练保存 `vecnormalize.pkl`；评估 callback 同步训练环境的 observation running statistics；ONNX 导出内嵌观测归一化。
- 原因：策略在归一化观测上训练，统计量不一致会让正确策略表现失真。
- 放弃方案：部署端重新估计统计量；评估环境使用初始或过期均值方差。
- 影响：模型和归一化文件是不可拆分的实验产物；warm start 优先恢复相邻 checkpoint 的统计量。
- 状态：已采用。

## D-005：保留可配置的 reach 奖励变体

- 日期：2026-10-12
- 决策：通过 `reward_type` 在 `origin` 与 `revised` 间切换；revised 在原奖励上加入 `-w_action_rate ||a_t-a_(t-1)||²`。
- 原因：在保留原实验可复现性的同时直接比较动作平滑惩罚，避免覆盖旧奖励或复制整个环境文件。
- 放弃方案：每次实验直接重写 `_reach_reward`；维护 `so101_reach_v1.py`、`v2.py` 等重复文件。
- 影响：新增奖励变体要加入明确 dispatch、配置字段、CLI choices、测试和 run 命名/记录。
- 状态：已采用；revised 可视化更平稳，但现有数据中成功率低于 origin。

## D-006：将 revised_v2 的成功奖励改为完成时一次性发放

- 日期：2026-10-13
- 决策：`revised_v2` 保留 distance/control/action-rate/velocity 结构，将 near 权重改为 `0.1`，只在距离小于 `0.02 m` 且连续保持 10 步后一次性加 `success_bonus_v2=20.0`。
- 原因：revised 的逐步 bonus 会奖励反复进入阈值但未连续保持成功的轨迹；实测失败轨迹可累计很高 bonus，甚至回报高于成功轨迹。
- 放弃方案：继续每个阈值内 step 支付 5 分；只提高平滑惩罚但保留奖励漏洞；此阶段同时修改 PPO 参数。
- 影响：reward 数值尺度和 mean reward 的可比性发生变化；需重新训练，并用成功率/距离/平滑度评价，不能只比较 mean reward。
- 状态：试验中；200 万步训练已完成，独立 final/best、多 seed 和平滑性评估仍待完成。

## D-007：区分瞬时阈值命中与 episode 成功

- 日期：2026-10-13
- 决策：`info["within_threshold"]` 表示当前步距离达标，`info["is_success"]` 只在连续保持完成并终止时为真，同时报告 `success_hold_count`。
- 原因：SB3 Monitor、EvalCallback 和后续评估把 `is_success` 当作 episode 成功指标；若它表示瞬时命中，会虚增成功率并掩盖抖动。
- 放弃方案：继续让 `is_success = (dist < threshold)`；只在分析脚本中猜测连续保持。
- 影响：成功率语义更严格；旧 run 和新 run 的 success 指标要注意代码版本差异。
- 状态：已采用。

## D-008：保留 8 个逻辑环境但用 DummyVecEnv 降低 WSL 内存

- 日期：2026-10-13
- 决策：`n_envs=8` 保持 rollout 规模和环境多样性，训练 venv 固定使用 `DummyVecEnv` 在一个 Python 进程中顺序推进。
- 原因：此前 SubprocVecEnv 多进程复制 MuJoCo/Python 状态，在 WSL 可用内存约 7.6 GiB 的条件下触发 OOM；改为单进程后训练可稳定运行，实际速度下降不明显。
- 放弃方案：直接把 `n_envs` 永久降为 4；继续使用 8 个子进程并仅依赖扩大 WSL 内存。
- 影响：环境 step 不再并行，但 PPO 每轮仍收集 `n_envs × n_steps` 样本；应持续记录 RSS 与 steps/s，并在硬件/配置变化后重新评估。
- 状态：已采用。

## D-009：先固定 PPO 参数，单独验证 reward

- 日期：2026-10-13
- 决策：引入 revised_v2 时不同时修改 PPO 超参数。
- 原因：控制变量，避免无法判断成功率变化来自 reward 还是优化器/采样设置。
- 放弃方案：同时降低 learning rate、epochs、clip range 并增加 target KL。
- 影响：若 revised_v2 解决奖励漏洞但训练仍不稳定，再单独开启 PPO 调优实验。
- 状态：已采用。

## D-010：训练产物同时保留 checkpoint、best、final 与配置

- 日期：既有设计，准确日期待确认
- 决策：周期保存模型和 VecNormalize checkpoint；保存 mean-reward best model、finally block 中的 final model/归一化文件以及完整 config。
- 原因：支持中断恢复、复现实验和 best/final 对比；异常中断也尽量留下 final 状态。
- 放弃方案：只保存训练最终模型；中断后从零开始。
- 影响：`--init-from` 是 warm start，一次调用的 `--timesteps` 表示本次继续训练的步数，不等于绝对累计步数；恢复时应配套加载 checkpoint 归一化文件。
- 状态：已采用；best 当前按 mean reward 而非 success rate 选择，后续需改进。

## D-011：将近期主线切换为 BDDL 课程 tutorial assignment

- 日期：2026-10-14
- 决策：在 2026-11-05 课程 slides 截止前，项目的首要目标改为完成 deep learning architectures 主题的教学型课程成果；现有 SO-101 reach 作为实验平台，完整 sim-to-real、实机和 pick-and-place 扩展暂缓。
- 原因：课程存在明确团队登记、提交和展示期限；现有仓库已经具备 PPO、可配置 MLP、评估和训练产物管理能力，适合承载架构实验而无需另建项目。
- 放弃方案：继续把主要时间投入 reward、SAC、DR 和实机闭环，再在截止前临时整理 slides；另建一个与现有实验脱节的课程代码仓库。
- 影响：近期代码工作只优先处理 SimBa/PPO 所需的架构接口、实验与图表；历史模型和未完成任务保留但降级，不删除产物。
- 状态：已采用。

## D-012：以 SimBa 解释深度 RL 的架构可扩展性

- 日期：2026-10-14
- 决策：课程主题归入 **Deep learning architectures**，暂定题目为 **Scaling Deep Reinforcement Learning: Why Bigger MLPs Fail and How SimBa Helps**；核心论文为 Lee et al., *SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning*（ICLR 2025）。
- 原因：该主题把网络规模、残差结构、归一化和 simplicity bias 与强化学习实践直接连接；论文同时给出 MLP 扩展失败、SimBa 组件、跨算法结果及 PPO 案例，适合 10 分钟 tutorial。
- 放弃方案：把主题写成宽泛的 sim-to-real 项目汇报；把 SAC/PPO 算法比较本身当作 deep learning architectures；在当前期限内转向尚无代码基础的 Transformer RL。
- 影响：slides 的中心论点是“更多参数只有在合适的归纳偏置和架构下才可扩展”，SO-101 实验用于验证这一论点，而不是宣称完整复现论文全部 benchmark。
- 状态：已采用；最终标题措辞仍可精简。

## D-013：用控制变量的 PPO 架构实验支撑课程结论

- 日期：2026-10-14
- 决策：最低实验矩阵比较当前 PPO MLP `2×256`、更大的标准 MLP 和 PPO+SimBa；固定 reach 环境、`revised_v2` 奖励、训练步数、seed、并行环境数、评估脚本和指标。优先比较参数量可比的网络，再进行 width scaling 或组件消融。
- 原因：若同时改变 reward、PPO 超参数、归一化、训练预算和网络结构，无法判断收益是否来自 SimBa。参数量可比与规模扫描分别回答“架构是否更好”和“架构是否更能扩展”。
- 放弃方案：只比较一次默认 PPO 与一次 SimBa run；直接引用论文的 Craftax 结果作为本项目证据；同时调 reward 与 PPO 超参数追求更高分数。
- 影响：正式实现前必须区分论文的网络内 RSNorm 与现有环境级 `VecNormalize`，记录参数量、wall-clock 和随机种子，并将单 seed 结果标为探索性结果。
- 状态：已采用。

## D-014：通过 MlpExtractor 扩展点接入 PPO+SimBa

- 日期：2026-10-16
- 决策：第一版 PPO+SimBa 保留 SB3 的 `RolloutBuffer`、`OnPolicyAlgorithm.collect_rollouts()`、`PPO.train()`、动作分布和 `ActorCriticPolicy` 公共推理/训练接口；在项目代码内实现兼容 `MlpExtractor` 契约的 SimBa actor/critic 主干，并通过 `ActorCriticPolicy` 薄子类覆写 `_build_mlp_extractor()` 完成接入，不直接修改已安装的 SB3 源码。
- 原因：当前源码核对表明 buffer 只保存 policy 生成的 rollout 数据并计算 GAE/returns，不拥有神经网络；`forward()`、`evaluate_actions()` 和 `predict_values()` 已通过同一个 `mlp_extractor`、action head 与 value head 保证采样和训练一致。只替换主干可最大限度保持 PPO 算法、数据格式和基线可比性。
- 放弃方案：为 SimBa 重写 rollout buffer；复制并修改整个 `PPO` 或 `train()`；仅覆写 `evaluate_actions()` 而让 rollout 收集继续使用另一套网络；直接修改虚拟环境的 `site-packages/stable_baselines3`。
- 影响：新增 extractor 必须提供 `latent_dim_pi`、`latent_dim_vf`、`forward()`、`forward_actor()` 和 `forward_critic()`，并验证 `forward/evaluate_actions/predict_values` 的 shape、梯度、一次 rollout/update、save/load 和导出影响。RSNorm、`VecNormalize`、Adam/AdamW 等不属于本决策的既定结论，仍需论文核对和控制变量实验。
- 状态：实现边界已采用；SimBa 代码、接口测试已完成，见 STATUS 的历史测试记录与 2026-10-20 收尾核验；不代表所有性能结论已验证。

## D-015：将 Fourier simplicity score 作为代理指标而非因果结论

- 日期：2026-10-16
- 决策：课程讲解、项目文档和后续实验报告必须把论文的 Fourier simplicity score 表述为“固定输入域与初始化协议下的低频函数倾向代理”。分别报告初始化阶段的 simplicity score、训练后的策略/价值函数行为和任务性能，不把二者的相关趋势写成已证明的因果关系。
- 原因：论文 Figure 4(a) 在二维合成输入网格上对未训练网络的 100 次随机初始化做 Fourier 分析，Figure 4(b) 则来自训练后的 DMC-Hard return；高频只是函数性质之一，且指标依赖输入坐标、尺度、采样域和 Fourier 定义。Residual、LayerNorm 与 RSNorm 还可能通过梯度传播、特征尺度和优化稳定性改善性能，不能把全部收益唯一归因于 simplicity bias。
- 放弃方案：把低频等同于普适意义上的简单；声称 SimBa 首次提出 RSNorm、残差连接或 LayerNorm；仅凭 Figure 4 的并列趋势宣称 simplicity bias 已被证明导致更高 return。
- 影响：slides 应明确区分“已有组件”“新组合与实验贡献”“作者解释”和“项目待验证假设”。SO-101 实验除任务回报外，还应在固定输入尺度下报告动作平滑性、扰动敏感性及必要的频率指标；是否复现论文 simplicity score 尚未决定。
- 状态：已采用为课程叙事与实验解释口径；不表示 SO-101 上的 SimBa 性能已经验证。

## D-016：第一版 PPO+SimBa 采用论文主干并复用 VecNormalize

- 日期：2026-10-16
- 决策：第一版 PPO+SimBa 使用独立 actor/critic 主干。actor 为 `Linear(feature_dim,256)` 后串行 1 个 SimBa block，critic 为 `Linear(feature_dim,512)` 后串行 2 个 SimBa block；每个 block 均为 `LayerNorm → Linear(d_h,4d_h) → ReLU → Linear(4d_h,d_h) → residual add`，stack 后再做 final LayerNorm。输入统一由现有环境级 `VecNormalize` 标准化，网络内不重复实现 RSNorm。
- 原因：上述宽度和 block 数来自论文 PPO+SimBa 配置，pre-LN residual、4× inverted bottleneck、ReLU 与 final LayerNorm 来自论文主体结构。当前 wrapper 已完整维护、冻结、同步和保存 observation running statistics；再加网络内 RSNorm 会引入双重归一化并污染主架构比较。
- 放弃方案：同时叠加 `VecNormalize` 与网络内 RSNorm；只为 SimBa 关闭 `VecNormalize` 而让 MLP 使用不同输入尺度；在第一版结构接入时同时修改 PPO optimizer、学习率和其他训练超参数。
- 影响：MLP 与 SimBa 主比较继续共享输入尺度，并都依赖匹配的 `vecnormalize.pkl`。本轮“严格按论文”指主干拓扑，不代表完整复现 Craftax PPO 训练配方；当前仍保留 SB3 初始化、单一 optimizer 和项目学习率。归一化或 optimizer 消融必须单独配置和报告。
- 状态：初版结构与历史接口/保存加载/ONNX 验证已采用；截至 2026-10-20 已有正式训练产物，后续 width 设置与性能验证边界见 D-018 和 STATUS 收尾节，不再表示尚未首次训练。

## D-017：训练预算消融采用冻结配置、同 seed 的独立从头训练

- 日期：2026-10-19；2026-10-20 核验产物。
- 决策：预算脚本按 200 万、300 万、500 万步串行启动独立 PPO 进程，使用同一份启动时冻结的配置；除预算及输出名称外不改环境/训练参数。失败或终端中断后不再启动后续项，dry-run 不创建配置/输出。
- 原因：区分训练预算与配置变化，不把项目 `init_from` 的 warm start 当成完整断点续训；避免源 YAML 修改污染后续 run。
- 影响：总预算为约 1,000 万步而不是续训至 500 万步；PPO 为完成 rollout 会略超目标。配置被冻结但代码未快照，运行期间不应修改代码。预览命令中的配置路径尚不存在，不应复制直接训练。
- 状态：代码及 8 项模拟测试已验证；既有批次保存步数达到目标。性能对照和固定条件 DR 汇总待验证。

## D-018：width 优化先缩 actor，再独立扩大 critic

- 日期：2026-10-20。
- 决策：先以 actor 128 / critic 512、blocks 1/2 对照原 actor 256 / critic 512；下一项固定 actor 128 和其他条件，仅把 critic 扩为 1024。不得把 width、深度、优化器与奖励同时改变后归因给单一因素。
- 原因：论文 Section 7.2 的 SAC 实验支持优先扩大 critic、谨慎扩大 actor，但不保证在 PPO reach 上复现；已有 128/512 run 的单 seed 内置名义表现改善，仍需独立 DR 与多 seed。
- 影响：当前用户 YAML 已设置 128/1024，bottleneck 512/4096；旧预算批次维持 256/512 快照不变。YAML 仍是 policy_arch=mlp，直接启动需显式启用 simba。不自动启动 1024 实验。
- 状态：128/512 配置、训练产物及内置名义结果已核验；128/1024 仅配置加载已核验，实际训练效果待验证。

## D-019：RSNorm 比较按统计生命周期解释，暂不改变 PPO 归一化

- 日期：2026-10-20；补充 D-013/D-016 中“网络内/外”表述。
- 决策：保留当前唯一的 VecNormalize observation normalization。作者公开 RSNorm 是 agent wrapper：在训练采样时更新累计统计，在网络更新时归一化原始 batch；不能把它等同于训练时使用当前 minibatch 统计的 BatchNorm，也不能声称 VecNormalize 与它工程上完全等价。
- 原因：论文 Section 7.1 的 replay 尺度不一致解释来自 SAC/off-policy；当前 PPO rollout 存的是采样时归一化 obs，并在更新时复用。直接用新统计量重归一化旧 obs 而保留旧 log_prob，可能改变概率比率和 KL 的含义。
- 影响：后续如比较 RSNorm，需单独设计统计冻结/更新及 old_log_prob 输入一致性；不双重归一化，不混入预算/width 实验。simplicity bias 不是时间低通滤波，“缺高频导致抗扰动差”与“延长训练必有涌现”均只是假设。
- 状态：论文、作者公开实现及本机 PPO/VecNormalize 源码已只读核对；解释口径已采用，替代归一化代码和因果性能实验未实施。
- 依据：论文 `https://arxiv.org/html/2410.09754v2#S7.SS1`；作者代码 `https://github.com/SonyResearch/simba/blob/master/scale_rl/agents/wrappers/normalization.py`。

## D-020：计划以本地 WSL 编辑、Git 同步、服务器执行缓解算力限制

- 日期：2026-10-20。
- 用户确定的方向：保留本地 WSL 为代码编辑和版本管理源，首次把代码、所需资产与环境迁移/复建到服务器，后续通过 Git 同步同一提交，实际实验在服务器运行。服务器不是 Windows 项目副本，也不作为另一套独立开发分支随意漂移。
- 原因：用户反馈本地 GPU 性能不足，并明确需要从 WSL SSH 入门开始指导；Codex 不假定能直接读写服务器，必要时由用户执行命令并反馈经脱敏的结果。
- 影响：先确认服务器访问与资源，再制定安装/同步方案。Git 不自动包含 outputs、权重、环境与所有资产；环境重建/整包迁移取决于 OS/架构、驱动和 CUDA 兼容性。禁止将密码/私钥提交到 Git；不自动 push、安装系统驱动或启动长任务。
- 与 D-001 的关系：当前已验证的本地可信源不变；未来服务器为用户指定的执行镜像，远程工作区规则、路径与验证流程待部署明确后再更新 AGENTS，不据此在未知服务器创建项目。
- 状态：仅工作流方向已确定；SSH、代码/资产上传、环境复制或重建、Git 同步、GPU/MuJoCo 验证及性能收益均待实施/待验证。本轮未连接服务器。

## Direct SSH Git synchronization (2026-10-20)

Decision supplement (D-020): GitHub is optional local backup; server code delivery uses direct SSH Git at user request. Use a bare receiver separate from the training checkout, with manual fast-forward update when training is inactive. No automatic deployment hook or proxy forwarding dependency. Receiver creation and remote validation are pending.


## D-021：课程结论先使用统一 final 协议，区分奖励历史与架构证据

- 日期：2026-10-22。
- 已采用的分析口径：origin MLP保留为历史起点；架构主比较使用revised_v2。将用户final 200回合结果标为单训练seed、无固定eval seed截图反馈，不把其优于旧MLP直接归因于SimBa架构。
- 原因：现有MLP为8env/无KL，候选SimBa为18env/KL0.33；best选择按nominal reward且与final VecNormalize可能错配。64/512 best在849456步，final在2009088步；其末期rollout不能等同early best的DR能力。
- 决策：报告优先比较同预算final+同run最终统计量；未来保存best时配套统计量，独立eval固定场景/seed并保存raw结果。历史文件保留，不以回退或估计统计量冒充精确best配对；验证集与最终测试集分离。
- 状态：证据分级和报告口径已采用；评估seed/导出/配对代码尚未实施，统一独立评估待验证。
- 补充D-018：本轮已找到并核验128/1024的2M/3M完整产物，不能再说仅配置存在；该组DR排名、多seed与机制仍待验证。
- 补充D-020：用户已完成server裸仓库、执行checkout、uv/Python3.11/torch2.8+cu126及smoke，来源为本对话输出及已提交环境记录；本轮未访问服务器。GitHub本地GUI、服务器命令行push/pull、outputs独立scp/rsync，沿用既有本地编辑/服务器执行规则。uv.lock、调度/配额与提速未验证。

## D-022：课程补充实验按主比较→分支单因素→可选扩展排序

- 日期：2026-10-22。
- 已采用的规划：先修评估、重评已有final，不优先继续扩大网络或训练预算。统一revised_v2、DR、2M、18env、n_steps512、batch1024、epochs18、lr3e-4、KL0.33与软件资产；训练seeds0/1/2，独立nominal/DR各≥200回合的固定测试集。
- 最低主比较为MLP2×256、MLP2×1440、SimBa64/512（blocks1/2）。2×1440按当前26维obs/5维action预估4,236,491参数，与SimBa4,252,235差约0.37%；必须实例化实测。该匹配控制总量，不控制分支分配、激活和初始化。
- 分支结论补SimBa256/512与64/256，形成固定critic的actor64→256及固定actor的critic256→512。128/512已有中间探索点；64/1024为可选峰值检验。五组各3seed共15次，最多复用两个协议一致seed0，至少新增13次2M；复用需先核对服务器版本。
- 精细容量匹配候选为MLP actor[172,172]/critic[2040,2040]，预估4,256,031参数；当前MLP构建器未支持独立列表，作为可选扩展，不把它写成已完成实验。
- 原因：已有actor sweep支持小actor方向，critic512→1024只在固定actor128的rollout上有单seed收益；尚无critic64/256→512或更大MLP受控数据，不能宣布普遍scaling律。128/1024约1696万参数，名义eval并不明显差，早best或不同预算会误导。
- 状态：优先级与候选矩阵已写入任务；没有新建YAML、自动运行或完成多seed结果。若大MLP不退化，照实报告；不以符合论文叙事为筛选标准。

## D-023：Fourier 分析分开测初始化偏置与训练后行为，不作单因素因果证明

- 日期：2026-10-22；补充D-015/D-019。
- 已采用的分析边界：性能证据与公平对照先完成，然后做A=随机初始化toy网络simplicity bias、B=训练后actor/critic局部频谱。A为架构/初始化的代理分布；B为具体模型/输入域的行为描述，不能互相替代。
- A以论文§5.1/Appendix B二维标量函数、[-100,100]^2、300×300网格和100初始化为参考，需先核对作者实现的频谱/初始化约定；SB3与论文初始化不同须单列。不是直接对高维checkpoint换成二维输入。资源预算先预览，不自动重计算。
- B用多个共同有效状态/目标切片，重算FK/TCP/target-TCP等关联量，分别分析actor确定性动作均值各维与PPO critic的V(s)。共同物理域+各自冻结归一化与共同归一化域的网络本体分析分别报告；clip饱和、DC、常量函数、幅度归一化、FFT边界泄漏与网格分辨率显式处理。
- 机制仅列假设：较简单控制律可能适合小actor，critic需拟合变化的价值目标；较大actor还可能改变PPO更新幅度。用共同轨迹的value/GAE误差、KL/梯度和扰动敏感性诊断，不能把SAC Q函数解释直接当成本项目PPO V函数证明，也不能把低频等同时间平滑或抗扰动能力。
- 状态：方法与解释边界已采用；Fourier代码、灰度/频谱图、100初始化统计和因果干预均未完成、待验证。
- 依据：https://arxiv.org/html/2410.09754v2#S5.SS1 、Appendix B与§7.2（SAC actor/critic scaling）。本项目复用VecNormalize且为PPO，只能称部分架构实践/改编分析，不能称完整复现论文全部实验。
