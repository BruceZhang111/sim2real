# 项目当前状态

**最后更新：2026-10-22（Asia/Shanghai）**

## 最新配方修订：8env / KL0.33（2026-10-22，用户指定）

- 本轮起始main工作区干净，未提交修改为空；未自动提交或推送，未访问服务器。用户指定MLP256/revised_v2为baseline，origin退出本轮评估和比较；历史文件保留。
- 覆盖先前18env计划：新五组统一8env、target_kl0.33、2M、seeds0/1/2；其他共同配方不变。8x512=4096 rollout batch，batch1024整除。
- 旧MLP/SimBa没有满足8env且KL0.33组合的已审计run，seed0均不能复用；五组需新训15次，共30M请求步。旧模型改配置或重评不会变成新配方模型。
- 本地更新reach.yaml的KL和reach_forSimBa.yaml的n_envs；其既有128/1024宽度及policy_arch保留。新增configs/architecture_8env内五份显式架构配置，用于正式矩阵，避免旧入口的隐式架构选择。
- docs/EVALUATION.md排除origin后保留8组历史revised_v2 final统一重评（16次/3200episode），只作探索证据；新五组完整final评估30次/6000episode。TASKS与D-024同步。
- 五组配置加载及共同字段一致性检查通过；真实26维obs/5维action的CPU实例化参数量：MLP256=146955、MLP1440=4236491、SimBa64/512=4252235、256/512=4751627、64/256=1095243。验证实例化仅用单环境，不采样/训练；正式YAML均为8env。
- 实际8env/KL0.33的4096步短流水线验证通过，产物outputs/smoke_arch8_kl033_20261022_173615；新增final配对清单已保存。21项架构/预算相关测试通过，git diff --check通过；没有启动2M正式训练或服务器任务，未重跑完整pytest/大规模评估。下一步仍是用户同步服务器后统一重评，再运行新矩阵。

## 评估实现收尾：固定评估与产物配对实现（2026-10-22）

- 开始时实际 WSL 仓库 main 跟踪 origin/main，git status/diff 干净；已读取 AGENTS、PROJECT、TASKS、原 2026-10-22 审计交接及 D-021～D-023。本轮全部源码/文档写入 WSL，未建立 Windows 副本，未 commit/push 或访问服务器。
- eval 默认 final、200 回合、eval_seed=20000；每回合显式 reset(seed=eval_seed+i)，确定性 CPU 推理、冻结 VecNormalize、原始 reward。禁止模型静默回退或归一化缺失时继续。相同模式内共享 seed，nominal/DR 为不同随机流分层，不宣称逐目标配对。
- 每次使用独立新目录输出 metadata.json、episodes.csv、results.json，记录逐回合 success、末距（m）、return、长度、时长与命令差分/幅度；JSON 记录初始目标，元数据含模型/归一化/配置 SHA256、代码版本/dirty/diff hash、依赖版本和 seed 口径。命令平滑性不等同实际关节或执行器平滑性。
- future best 保存同一时点 VecNormalize 和 best_pair.json；future final 保存 final_pair.json。评估验证存在的哈希清单；历史 final 仅标记 run-relative legacy pairing，不能追溯证明同时保存。历史 best 缺 stats/清单直接报错，未伪造或覆盖历史文件。
- 修改 sim2real/eval.py、sim2real/train.py；新增 tests/test_eval_protocol.py、docs/EVALUATION.md（9 个 run 清单及服务器命令）。现有审计结论和单训练 seed/n_envs/KL 混杂不变；短验证不作为课程性能证据。
- 验证：新增9项评估测试 + 13项 SimBa + 8项预算脚本，共30项通过。真实历史 MLP 的 nominal/DR 各2回合重复评估逐回合完全一致（共8回合），不修改归一化原文件。覆盖严格配对、哈希篡改拒绝、导出防覆盖、best回调和seed边界。
- make smoke 通过（4000请求/4096实际步）；补 final manifest 后在独立目录再运行等价 smoke，通过，目录 outputs/smoke_eval_pairs_20261022_165730。该新smoke的best nominal、final DR各1回合CLI评估和哈希验证/原始导出也已贯通。只新增本轮短 smoke 产物，保留全部历史模型。未运行完整 pytest；本轮相关30项已覆盖评估、架构接口和串行预算入口。既有 ONNX 弃用提示及测试用 Pendulum 无 Monitor 提示不影响通过结果。
- 待办：用户审阅并提交本地修改后 git push server main；确认服务器训练停止、工作区干净再 git pull --ff-only，核对 SHA/依赖/资产后依 docs/EVALUATION.md 重评 9 组 final nominal/DR 各200回合。尚未同步或重评服务器，不宣称性能已补齐。随后冻结五组3seed配方，再进行 Fourier/有效状态切片分析。

## 历史收尾：实验审计与课程最小补充方案（2026-10-22）

本节为当时的审计交接；最新实现见上节。后文 2026-10-20 及更早内容为历史快照。早期“未训练”“服务器待配置”等状态不再代表当前状态，但原有有效证据保留。课程目标/范围未变化，本轮不修改 PROJECT.md 或 AGENTS.md。

### 范围、证据与验证边界

- 已通过 WSL 确认外层目录 `/home/brucezhang/Dev/Projects/SO101 Sim2Real`；实际 Git 根目录、代码与 docs 均在其下 `sim2real/`。开始时 main 跟踪 origin/main，HEAD 为 `2e503cc`，工作树干净。
- 本轮只读扫描全部 outputs：发现 9 个 reach run 和 1 个 pickplace run 具备 config.yaml/final_model.zip；课程比较仅使用 reach。读取 YAML、模型 zip/data、policy.pth（CPU、weights_only=True）、eval/evaluations.npz 和 TensorBoard 原始 scalar；没有启动训练、仿真、独立 eval 或 Fourier 实验，也未修改代码/配置/历史产物。
- 当前本地预算批次只发现 ts5000000 的完整 run；此前文档中已核验的 ts2000000/ts3000000 本轮未在该目录找到，不推测是否删除/移动，也不把历史核验当作本轮重新验证。
- 未发现本地 sim2real.train/预算脚本进程（检查时刻快照）；没有访问服务器或确认服务器当前负载。
- 下表参数量来自 final 权重张量 numel；actor 包含 action head 与 log_std，critic 包含 value head。曲线数值为原始最后 10 个日志点的算术均值，非 TensorBoard 平滑值；rollout 指标本身已是滑动 episode 统计，不能当成独立重复实验。

### 实验清点与已有证据

| run（outputs/ 下） | 架构/宽度 | n_envs / target_kl | final 步数 | 参数量 | 末10 rollout 成功率 | 末10 内置 nominal eval 成功率 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ppo_reach_20261012_114650_nenvs8_origin | MLP 2×256，origin reward | 8 / None | 2,002,944 | 146,955 | 0.655 | 0.990 |
| ppo_reach_20261013_120248-revised_v2 | MLP 2×256，revised_v2 | 8 / None | 2,002,944 | 146,955 | 0.849 | 0.945 |
| ppo_reach_20261019_110612-simba-L12 | SimBa 256/512，blocks 1/2 | 8 / None | 2,002,944 | 4,751,627 | 0.330 | 0.750 |
| ppo_reach_20261019_174807-KL0.33 | SimBa 256/512，blocks 1/2 | 18 / 0.33 | 2,009,088 | 4,751,627 | 0.434 | 0.925 |
| ppo_reach_20261020_104421-actor128-critic512 | SimBa 128/512，blocks 1/2 | 18 / 0.33 | 2,009,088 | 4,353,163 | 0.577 | 0.980 |
| ppo_reach_20261020_113536-actor128-critic1024 | SimBa 128/1024，blocks 1/2 | 18 / 0.33 | 2,009,088 | 16,958,603 | 0.828 | 0.985 |
| ppo_reach_20261020_154940-actor128-critic1024-timesteps | SimBa 128/1024，3M | 18 / 0.33 | 3,004,416 | 16,958,603 | 0.927 | 0.980 |
| ppo_reach_20261021_203622-actor64-critic512 | SimBa 64/512，blocks 1/2 | 18 / 0.33 | 2,009,088 | 4,252,235 | 0.933 | 0.985 |
| timestep_sweep_20261019_222032_a097e2ee/ppo_reach-simba-ts5000000 | SimBa 256/512，5M | 18 / 0.33 | 5,004,288 | 4,751,627 | 0.688 | 0.940 |

- 上述 run 的训练 seed 均为 0。除 origin 外 reward 均为 revised_v2；各 reach 配置的 dr 段相同且 enabled=true。
- origin 是历史 MLP 起点，不能与 revised_v2 直接比较 reward 数值，也不能把奖励改进归功于架构。现有直接架构基线应使用 revised_v2 MLP；origin 可作为“奖励设计→稳定基线”的历史背景。早期成功语义/代码版本与当前 eval 是否完全一致仍待核对。
- 固定 critic=512 的 256→128→64 actor sweep，在 YAML 层面其余环境/训练条件相同，支持“缩小 actor 在该 seed/配方下改善训练表现”。固定 actor=128 的 critic 512→1024，支持训练 rollout 改善；不等于已证明 critic 越大越好。128/1024 的内置名义指标并不明显弱于 64/512；前者 DR 鲁棒性、时间效率与多 seed 排名待验证。
- 2M 与 3M 的 128/1024 run：evaluations.npz 的 timesteps/results/ep_lengths/successes 在前 2M 的数组前缀完全相同。它们不能算作两个独立 seed；不据此前缀推断启动方式或严格恢复语义。
- 64/512 仍比 MLP 2×256 大约 28.9 倍；其 actor 为 35,402、critic 为 4,216,833 参数（约 99.2% 在 critic）。128/1024 的 critic 为 16,822,273 参数。报告不能只报 hidden width 或声称 SimBa 更省总参数。
- TensorBoard event 首末时间跨越不同机器/运行条件，含潜在暂停或启动开销；本轮不据此声称服务器提速或架构计算效率结论。

### 用户服务器 final 评估反馈（2026-10-22，非本轮重跑）

来源为本对话终端截图，200 episodes、deterministic=True，模型均为 final_model.zip，配套各 run 根目录 vecnormalize.pkl。截图未提供固定 eval seed/per-episode 原始数据；本地仅找到训练回调 evaluations.npz，不能把它当作下面的独立 DR 评估原始数据。

| 模型 | nominal 成功率 | nominal 距离均值/中位数 mm | nominal return | DR 成功率 | DR 距离均值/中位数 mm | DR return |
| --- | --- | --- | ---: | --- | --- | ---: |
| MLP revised_v2 | 94.0%（188/200） | 13.4 / 12.0 | 18.09 | 86.0%（172/200） | 15.5 / 13.2 | 16.40 |
| SimBa 64/512 | 98.5%（197/200） | 11.2 / 10.4 | 18.71 | 96.0%（192/200） | 12.1 / 11.3 | 18.40 |

- 支持本次用户评估下 SimBa 的 nominal +4.5 个百分点、DR +10 个百分点，DR 平均末距降低 3.4 mm；这是单训练 seed 的候选优势，不是多 seed 显著性结论，也不是 origin 对照的结果。
- MLP 为 8 envs/无 target_kl，而候选为 18 envs/target_kl=0.33，且训练环境依赖版本可能不同。n_steps=512 时 rollout batch 分别为 4096/9216，更新节奏也不同；不能把该结果全归因于架构。

### best/final 与评估可信度

- 已核对 best 元数据及 evaluations.npz：64/512 best 为 849,456 步、nominal mean reward 19.5688017；revised_v2 MLP best 为 1,925,000 步、19.68347615。128/1024 的 2M 与 3M best 都在 649,584 步（19.912 左右）。
- rollout 用训练中随机动作，内置 eval 用 nominal 环境和确定性动作；best 按 nominal mean reward 的严格新高保存，与末期 rollout/DR 成功率不是同一个选择标准。平滑曲线峰值不能代替原始评估点。
- train.py 保存 best 时不另存对应 VecNormalize；eval.py 默认 best 却总加载根目录 final 统计量。best/统计量错配风险仍存在且未修复；不能再用“early best + final stats”的劣势证明大网络泛化更差。
- 课程主比较优先 final+同一 run 的 final VecNormalize、相同训练预算和固定评估场景。后续补 best 统计量保存/严格配对，对缺失配对的历史 best 明确标记，不伪造或覆盖原始产物。
- eval.py 尚无显式 eval seed、逐 episode CSV/JSON、平均长度/平滑性输出；没有固定测试场景。先完善评估，再运行新训练；本轮仅制定方案，未实现。

### 课程实验的最小补充顺序（计划，均待验证）

统一配方先冻结为 revised_v2、DR 当前范围、2M 环境步、18 envs、n_steps=512、batch_size=1024、n_epochs=18、lr=3e-4、target_kl=0.33、相同归一化/软件/资产版本。所有组统一使用，不把此配方描述成最优 PPO 配方。训练 seeds 至少 0/1/2；独立 eval 固定与训练不同的场景 seed 集合，nominal/DR 各至少 200 回合、冻结统计量、确定性动作。评估 seeds 与训练 seeds 分开记录。

1. **P0：不新增训练，先重评已有 final。** 评估 MLP origin（单独 reward 背景）、MLP revised_v2、SimBa 256/512、128/512、64/512、128/1024（2M/3M）和 256/512 5M；保存逐 episode 原始结果、长度、距离与动作平滑性。DR 分项扰动（只延迟/只噪声/只动力学）优先挑 64/512、256/512 做诊断，非全量笛卡尔网格。
2. **P1：主结论最小三组，每组 3 个训练 seed。** A=MLP [256,256]；B=MLP [1440,1440]（预估 4,236,491 参数，与 64/512 总参数差约 0.37%，创建后须实测）；C=SimBa actor64/critic512、blocks1/2。A/B 按统一配方新跑 seed0/1/2；C 的 seed0 可作为候选复用，须先确认服务器代码/配置/软件版本，不满足则重跑。最多复用 1 项，因此至少新增 8 次 2M 训练（约 16M 环境步）。
3. **P2：若要讲 actor/critic 不对称，加入 D=SimBa256/512 与 E=SimBa64/256，每组 seed0/1/2。** D 已有 seed0，仍须协议核对；E 全部待跑。与 C 形成固定 critic 的 actor64→256、固定 actor 的 critic256→512。五组完整最低矩阵为 15 次训练，最多复用 C/D 两个 seed0，新增至少 13 次（26M 环境步）；不声称已经做完。
4. **可选峰值检验：SimBa64/1024。** 先统一 seed0 预览，若要进入结论同样补足三 seed；它固定 actor64 后检验 critic512→1024 是否饱和或退化。actor32/512、深度扫描和组件消融排在上述证据之后，不再同时扫学习率/epochs/DR 强度。
5. 参数量匹配只解决容量混杂，B/C 的 actor/critic 分配仍不同。若核心结论强调精确架构贡献，可扩展 MLP 独立 net_arch，再做 actor [172,172] / critic [2040,2040]（预估总 4,256,031；分支数接近 35,402/4,216,833）对照。当前 build_model 的 MLP 接口只接受共享宽度列表；该接口扩展与对照均未实现，不作为最低交付硬前置。
6. 输出三类图：统一 final 的 nominal/DR 成功率及距离（每 seed 点+均值/训练 seed 方差）、训练曲线/样本效率、参数量与计算成本。episode 重采样误差不能代替训练 seed 方差；3 个 seed 仅为课程最低重复数，仍应注明统计局限。尚未得到 bigger MLP 退化的受控结果；若不退化，如实报告而不筛选失败 seed。

### actor/critic 机制解释：候选假设与诊断

- PPO actor 输出动作分布均值；critic 估计 V(s)，并经 TD residual/GAE 影响优势，不是 SAC 的 Q(s,a)。任务策略可能比随机动力学下的价值预测更简单，因此小 actor/较大 critic 是合理假设，不是本轮因果结论。
- 固定 critic512 时 actor256/128/64 的末10 approx_kl 约 0.569/0.240/0.078；支持“actor 宽度可能影响更新幅度”的诊断方向。target_kl 是停止条件而非 KL 硬上限；不能把相关性解释成已证明机制。
- 固定 actor128 时 critic512/1024 的末10 explained_variance 约 0.440/0.461；64/512 为约 0.358，而旧 MLP 为约 0.714。说明高 explained_variance 并不自动对应更好策略，跨 run 的不同状态分布也不可直接等价比较。
- 待验证：在共同的留出轨迹上比较 value 预测误差/EV（注意策略特定的 V 目标）、GAE/TD residual 分布、actor 梯度/KL/clip_fraction、动作差分与局部输入扰动敏感性；需要相同观测分布、固定目标定义，不直接比较不同归一化回报尺度的 value_loss。critic 噪声拟合、塑性/优化限制和输入敏感性均仅是候选解释。

### 下一阶段 Fourier / Simplicity Bias 分析协议（尚未实现）

- 两层问题分开：A=随机初始化架构的 simplicity-bias 代理；B=训练后 actor/critic 的函数频谱和局部敏感性。灰度函数图与 FFT 频谱图各司其职，不能仅凭一张平滑灰度图断言泛化原因。
- A 参考论文 §5.1/Appendix B：二维输入、标量输出、[-100,100]^2 的 300×300 网格，100 次随机初始化。先做少量 seed/低分辨率资源预览，再批量计算；核对作者 Fourier 实现、幅值/功率、频率权重和初始化方案后才能称复现。比较实际 SB3 初始化与论文初始化时单独标注，不能混入架构结论。
- A 不直接把高维 checkpoint 填进二维输入；明确建立相同标量读出/初始化协议的二维 toy 网络，或把项目切片方法标为“改编”。actor/critic 分别分析，不用宽度相同假设参数相同。
- B 优先从共同留出有效状态采样，固定姿态与 last_action，扫描目标平面时重算 target-TCP 等关联观测；如改变关节也重算 FK，避免独立乱改 qpos/TCP/误差导致离开物理观测流形。每个模型各用其配套冻结 VecNormalize；actor 分动作维度分析确定性均值（主图尽量避免 clip 饱和），critic 分析标量 V，不分析随机动作样本。
- 网络本体用共同归一化输入域，部署行为用共同原始物理域+各自归一化，两者分别报告；归一化/clip 对谱的影响需显式说明。不要将论文 [-100,100] toy 域直接当作项目 clipped-to-10 的有效观测域。
- 至少输出共享坐标/色标的函数灰度图、log FFT 幅值或功率图、径向频谱、按明确约定计算的频率加权复杂度/高频占比。常量/近常量输出、DC 是否去除、窗函数/边界泄漏、网格分辨率和读出尺度需显式处理；未核对作者实现前不得称数值完全复现，也不把常量零策略误判为好策略。
- 初始化多 seed 的频谱分布描述架构偏好，训练后多状态切片描述学得行为；两者与成功率只有相关性，PPO 优化、样本分布和探索仍是替代解释。
- 原文依据：https://arxiv.org/html/2410.09754v2#S7.SS2 （actor/critic scaling 为 SAC 实验）；https://arxiv.org/html/2410.09754v2#S5.SS1 与 Appendix B（初始化 Fourier）。本项目为 PPO，不能写成完整复现 SAC 扩展规律。

### 服务器工作流现状（来自本对话反馈与已提交清单）

- 用户已完成 VS Code Remote-SSH、server bare receiver `/home/zyg/git/sim2real.git`、执行目录 `/home/zyg/Projects/sim2real`；SSH `zyg@10.181.213.48:3389`。本地推送 `git push server main`，服务器训练停止/结束后 `git pull --ff-only`；GitHub 推送独立由本地 GUI 管理。
- 已有用户输出验证 uv0.13.3、Python3.11.15、torch2.8.0+cu126、CUDA可用/2卡、依赖检查及 smoke 成功；环境清单已提交。本轮未重跑；已提交 requirements-installed.txt 含 rich==15.0.0、未列 tqdm，不能当作当前完整环境。requirements.txt 还有重复 rich 声明（>=13.0 与 >=13.7,<15），与该快照不一致；pyproject.toml 尚未声明 tqdm。仅记录问题，本轮未改依赖；重新核对服务器 freeze、统一两份声明及锁定环境仍待执行。未生成 uv.lock，不得写成已锁定环境。
- outputs 不经 Git，用 scp/rsync 在本地 WSL 与服务器传输。交互 GLFW 在无 DISPLAY 的 SSH 上失败符合 headless 限制；可下载配套 run 本地可视化或用已有视频入口。GPU 配额/调度/断线保持、可复现实测吞吐仍待确认。

### 本轮收尾与恢复入口

- 本轮修改仅为 STATUS.md、TASKS.md、DECISIONS.md；无业务代码/配置变更，无新训练或测试通过声明。完成数据一致性只读检查和文档 diff 检查，未自动 commit/push。
- 下一轮首要任务：在本地完善固定 eval seed、逐 episode 导出和模型/统计量配对，保留历史产物；同步到服务器后统一重评已有 final，再冻结五组最小矩阵。不要先重训所有大网络，也不要先凭频谱解释结果。

供新对话恢复提示词：

> 请继续 WSL Ubuntu 项目。先确认外层 /home/brucezhang/Dev/Projects/SO101 Sim2Real，实际 Git 根目录为其下 sim2real；读取 AGENTS.md、docs/PROJECT.md、STATUS.md 顶部 2026-10-22 收尾、TASKS.md、DECISIONS.md D-021～D-023 并检查 git status/diff。不要在 Windows 建副本，不自动 commit/push/启动长训练。已审计 9 个 reach run；原始 origin MLP 仅作奖励历史，主基线为 revised_v2 MLP。用户服务器 final 各200回合：MLP nominal94%/DR86%，SimBa64/512 nominal98.5%/DR96%，属单seed且 n_envs/KL 混杂。best 过早且配 final 统计量的风险未修复，eval 尚无固定seed/原始导出。先在本地修复评估协议并做针对性验证，再由用户同步服务器重评 final；随后统一18env/KL0.33/2M/seeds0,1,2，最低五组 MLP256、总参数匹配MLP1440、SimBa64/512、256/512、64/256，最多复用两个seed0（版本待核对）。课程目标是可靠架构/分支容量证据，再做初始化 toy Fourier 与训练后有效状态切片分析；严禁将灰度平滑当因果证明。服务器路径 /home/zyg/Projects/sim2real，裸仓库 /home/zyg/git/sim2real.git，SSH3389；本地修改和提交，服务器仅拉取执行。保留所有历史产物与未提交修改。

## 历史收尾：已核验现状与下一轮交接（2026-10-20）

本节保留 2026-10-20 当时的核验记录；最新状态见顶部 2026-10-22 收尾。

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

## Direct SSH Git synchronization (2026-10-20)

User selected direct WSL Git -> SSH -> server after GitHub network problems and opted out of proxy forwarding. Local remote server prepared as ssh://zyg@10.181.213.48:3389/home/zyg/git/sim2real.git, preserving origin and upstream. Proposed bare receiver /home/zyg/git/sim2real.git is pending user creation; execution path /home/zyg/Projects/sim2real approved by user. No remote login/push/clone was performed by AI. uv downloads/lock and driver compatibility remain unresolved. See rewritten MIGRATION.md.

## Direct SSH checkout verified by user (2026-10-20)

User supplied: execution cwd /home/zyg/Projects/sim2real, clean main tracking origin/main, origin /home/zyg/git/sim2real.git, HEAD b85adacdc2bb31c7696044703ceb709a7198b058 matching local WSL. Fast-forward-only pull configuration command completed without reported error. Direct SSH receiver/push/checkout workflow is verified by user outputs; AI has not logged into the server. Local SB3 2.9.0 installed metadata requires torch>=2.8,<3.0. Therefore selecting an older CUDA 11.8 torch build alone may conflict; do not silently downgrade SB3 or assert equivalence. Next: install user-scoped uv through an available download route, verify package index reachability, then resolve and test a driver-compatible stack locally. uv.lock and remote Python3.11 environment are not yet created.

## uv bootstrap without server pip (2026-10-20)

User reports /usr/bin/python3 has no pip. Prepared official PyPI uv 0.13.3 manylinux x86_64 wheel locally; verified wheel SHA256 against PyPI and extracted the standalone uv binary. Local --version succeeded. Ignored transfer files are under outputs/migration-tools/uv-0.13.3-linux-x86_64/; provenance in docs/environment/server-20261020/uv-bootstrap.json. User must transfer binary/checksum over SSH, validate and install in ~/.local/bin. No server install has been performed. No system pip or sudo required. Python3.11 download and project dependencies remain pending.

## Server uv installed (2026-10-20)

User verified transferred binary checksum and installed uv 0.13.3 in /home/zyg/.local/bin without pip/sudo. Python 3.11.15 installation and dependency downloads remain pending. Local read-only official cu126 index inspection found torch 2.14.0+cu126 cp311/Linux x86_64 wheels, allowing a same-PyTorch-version candidate; driver minor-compatibility constraints and actual CUDA execution still require validation. No dependency changes or GPU training performed.
