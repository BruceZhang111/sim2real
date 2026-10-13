# 可执行任务清单

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 已完成 · `[!]` 被阻塞

## P0：保护当前实验并得出 reward 结论

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

## P1：提高实验可信度

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
- [ ] 更新 README 中测试数量或取消易过时的硬编码数字。
  - 验证：README 与实际收集到的测试一致。

## P2：在 reward 固定后调 PPO

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

## P3：sim-to-real 验证

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

## P4：课程大作业与前沿算法

- [ ] 明确课程评分要求、截止日期、报告/演讲格式和可用算力。
  - 验证：形成一页实验约束说明。
- [ ] 选择一项与深度学习强相关、能在本项目中形成公平实验的前沿方向。
  - 候选：Transformer/sequence-model policy、Decision Transformer 类离线 RL，或现代 model-based RL；具体算法及 2025–2026 论文待检索确认。
- [ ] 先构建最小可比较接口，再实现算法。
  - 验证：新方法复用同一 env、观测、动作、seed 和 eval 指标；pipeline 变化与算法变化分开记录。
- [ ] 形成课程交付物。
  - 验证：论文综述、方法图、可复现实验配置、结果表/曲线、失败分析和现场演示材料齐全。
