# Public release candidate — 2026-08-05

## 发布内容

- A/B/C/D/E/F 材料学原理发现与对照验证控制器。
- X/Y/Z/W 经验利用、候选审计和 MatterGen 生成闭环。
- 完整 Python 源码、4 组回归测试、MatterGen 适配器和便携模板策略。
- evaluator 实际读取的 1000 条结构及其聚合结果。
- 15 条通过验证的原理、45 组关键对照实验、1000 条逐项 `E_hull` / S.U.N. 数据。
- 架构、结果、复现、代码数据清单和发布前检查文档。

## 结果摘要

- in-sample strict S.U.N.: `344/1000 = 0.344`。
- `E_hull < 0.03`: `592/1000`；`E_hull < 0.10`: `884/1000`。
- 1000 个不同约化化学式，evaluator 报告 100% 成分与结构新颖性。
- evaluator 几何口径下仅 37 个结构有效，其中 10 个同时满足 `E_hull < 0`。

## 相对内部快照的发布处理

- 仅清理个人绝对路径、内部中转站地址和集群默认值；核心算法未改写。
- 历史 elite replay 路径保留在 provenance 策略中，但公开的模板策略关闭了不可复现的 replay。
- 不发布真实凭据、内部日志、大体积 evaluator 数据、MatterGen 权重或旧结果快照。
- 项目自有代码以 Apache License 2.0 发布；第三方组件和外部数据仍遵循各自许可条款。

## 已知限制

- 当前结果只完成 in-sample evaluator 评估，尚无 out-of-sample、DFT 或实验确认。
- 最终化学空间高度集中于 Br/Rb 体系。
- evaluator 的 `structural_validity` 与 oracle `validity_rate` 口径不同，不能只报告后者。
- 冻结 `input.json` 缺少 `sites[*].properties` 空字典；Pymatgen 与本次 evaluator 可读取，但严格 schema 消费者可能要求补齐。
- 完整动态轨迹依赖外部 LLM、MatterGen、checkpoint、评估数据和计算资源，不能只靠本精简包逐字节重放。
