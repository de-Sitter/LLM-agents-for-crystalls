# 系统架构

## 设计目标

系统不让 LLM 直接承担所有晶体坐标构造，而是把科学推理、实验设计、结构生成和数值评估拆成相互审计的层：

1. 从已有结果提出可检验的材料学机制。
2. 用 primary/control 对照而非单个候选验证机制。
3. 将自然语言意图翻译成程序化模板或 MatterGen 请求。
4. 由统一 evaluator 计算 `E_hull`、唯一性、新颖性和 S.U.N.。
5. 同时保存成功原理、失败边界和不能迁移的条件。

## A/B、C/D、E/F：原理发现环

| 智能体 | 职责 | 主要输出 |
|---|---|---|
| A | 提出微观机制与适用范围 | mechanism proposal |
| B | 查找反例、混淆因素并给出反提案 | audited mechanism / counterproposal |
| C | 把机制转成可证伪 primary/control 预测 | comparison design |
| D | 审计因果变量、对照匹配与可执行性 | accepted/revised prediction |
| E | 用 MP pool、模板生成器或 MatterGen 材料化 | executable bundles |
| F | 审计 bundle 是否忠实、可运行且未偷换目标 | execution consensus |

入口：`src/crystal_llm/run_material_physics_mvp.py`。

当前公开结果中，主状态最终保留 15 条 `validated_principle`；关键验证证据整理为 45 组 bundle。完整逐项数据见 `artifacts/intermediate_data.md`，精简可机读状态见 `artifacts/af_principles_state.json`。

## X/Y、Z/W：经验利用与生成环

| 智能体 | 职责 | 主要输出 |
|---|---|---|
| X | 读取原理书和历史反馈，提出材料候选队列 | candidate descriptions |
| Y | 检查是否越过失败边界，并给出修复/反提案 | material consensus |
| Z | 把材料描述翻译为模板参数或 MatterGen request | executable generation request |
| W | 审计电荷、元素集合、过滤条件和生成契约 | locked request |

入口：`src/crystal_llm/run_xy_experience_debate.py`。

最终输入由累计合并形成：先冻结 452 个 base 结构，再加入 548 个经评估、去重的 dynamic 结构，得到 1000 个不同约化化学式。逐项来源索引保存在 `artifacts/dynamic_cumulative_report.json`。

## MatterGen 的位置

MatterGen 是结构生成后端，不是稳定性裁判。系统向它提供元素体系和 `energy_above_hull` 等条件，随后：

1. MatterGen 在原子类型、分数坐标和晶格空间生成结构。
2. `mattergen_backend_prototype/mattergen_adapter.py` 转为 Pymatgen MSON。
3. 适配器执行站点数、体积、化学系统和重复结构过滤。
4. evaluator 独立计算最终 `E_hull` 和 S.U.N.。

因此，生成条件中的目标能量不能替代下游 evaluator 结果。

## 关键实现映射

| 模块 | 作用 |
|---|---|
| `material_physics_schema.py` | mechanism/prediction/execution 数据约束 |
| `local_agent_runtime.py` | 本地 agent 工具循环和结构化交互 |
| `llm_client.py` | Responses API、重试、SSE/兼容处理 |
| `run_material_physics_mvp.py` | 六 agent 辩论、材料化、评估和 postmortem |
| `run_xy_experience_debate.py` | 四 agent 候选搜索、MatterGen 和 sequential memory |
| `templates.py`, `generate.py` | 可解释程序化晶体模板和生成入口 |
| `filters.py`, `chemistry.py` | 化学式、氧化态、几何和去重过滤 |
| `analyze_evaluator_run.py` | evaluator 日志到逐结构结果的整理 |
