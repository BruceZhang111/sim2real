# SO-101 Sim2Real 工作规范

## 项目与工作区

- 本仓库用于 SO-101 / SO-ARM100（Feetech STS3215）机械臂的强化学习 sim-to-real：在 MuJoCo 中训练，在仿真中验证后再导出并部署到实机。
- 唯一可信工作区是 WSL Ubuntu 中的 Git 仓库：`/home/brucezhang/Dev/Projects/SO101 Sim2Real/sim2real`。不得在 Windows 路径创建或维护项目副本。
- 开始修改前必须先执行 `pwd`、查看项目结构和 `git status --short --branch`，确认实际位于上述仓库，并保留用户已有的未提交修改。
- 不确定的项目事实不得猜测；在文档中写成“待确认”，在代码修改中先向用户核实。

## 稳定技术约束

- Python 要求 `>=3.10`；项目说明和依赖注释记录的主要环境为 conda 环境 `sim2real`、Python 3.11。
- 核心栈为 MuJoCo 3、Gymnasium、Stable-Baselines3、PyTorch、NumPy；PPO 是 reach 的默认算法，SAC 也受支持。
- 配置以 `configs/*.yaml` 为实验入口，字段定义以 `sim2real/config.py` 为准。未知配置键必须报错；不要在业务代码中复制一套隐藏参数。
- Python 命名使用 `snake_case`，类使用 `PascalCase`；新增公共函数和复杂逻辑应写类型标注及说明“为什么”的简短注释。
- 保持 Gymnasium API：`step()` 返回 `(obs, reward, terminated, truncated, info)`；`info["is_success"]` 只表示完整任务成功，不得与瞬时条件混用。
- 奖励实验应通过明确的 `reward_type` 和独立参数切换；保留既有变体以保证旧实验可复现，不要用注释/复制 Python 文件切换奖励。

## 数据、单位和图表

- 长度统一使用米，角度使用弧度，速度使用弧度/秒，时间使用秒，频率使用 Hz；对外展示可换算为 mm，但变量名或图表轴必须标单位。
- 策略动作保持 `float32`、归一化范围 `[-1, 1]`；观测数组保持一维 `float32`，维度变化必须同步 observation space、部署端和测试。
- 训练曲线至少区分 run 名、reward 类型、seed、模型（best/final）、是否启用 domain randomization；不同实验比较时保持横轴 timesteps 和评估条件一致。
- 保存并使用与模型配套的 `config.yaml` 和 `vecnormalize.pkl`；不得拿一组归一化统计量配另一模型。

## 测试与验证

- 修改配置/奖励/环境后，最低验证顺序为：静态语法或配置加载、针对性单元测试、`make smoke`、固定 seed 的名义动力学评估、开启 domain randomization 的评估、最后才是实机。
- 长训练正在运行时，不启动会显著争用 CPU/GPU/内存的测试、仿真或可视化；只做轻量只读检查和小型文档操作。
- reach 成功必须满足距离阈值并连续保持 `success_hold_steps`；报告中同时记录成功率、末端距离、episode 长度和稳定/抖动表现，不能只看 mean reward。
- 奖励修改后要检查各分项量级、失败 episode 是否可能获得高于成功 episode 的回报，以及 `is_success` 是否只在真正终止成功时为真。
- 实机前必须先使用 MuJoCo stand-in 走完整部署路径；首次实机运行采用低 `max_step_rad`、短步数，并保留断电/急停条件。

## 受保护内容

- `assets/so101/so101_new_calib.xml` 及上游 vendored 模型/网格默认视为只读；更新应使用 `scripts/fetch_assets.sh`，不要手工改上游资产。
- 不得随意删除 `outputs/` 中仍用于比较或恢复训练的模型、checkpoint、TensorBoard 日志、配置快照和归一化统计量。
- 不得随意删除或覆盖校准参数、部署配置、Git 历史、用户未提交修改，以及正在运行的训练所写入的目录。
- `Makefile` 的 `clean` 会递归删除训练输出和缓存，除非用户明确授权且已核对目标，否则禁止执行。

## 每轮任务收尾

1. 复查 `git diff --check` 和 `git status --short --branch`，明确区分本轮修改与原有修改。
2. 运行与风险匹配且不会干扰现有训练的验证；未运行的检查及原因必须说明。
3. 业务状态有变化时更新 `docs/STATUS.md`；任务状态变化时更新 `docs/TASKS.md`；形成长期技术取舍时追加 `docs/DECISIONS.md`。
4. 向用户列出修改文件、验证结果、遗留风险和可执行的下一步；不得把“未验证”描述成“已完成”。
