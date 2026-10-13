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
- 状态：试验中；200 万步训练正在进行，最终效果待确认。

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
