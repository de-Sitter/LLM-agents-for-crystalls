# v1.0.0 — 2026-08-05

## 发布内容

- A/B/C/D/E/F 材料学原理发现与对照验证控制器。
- X/Y/Z/W 经验利用、候选审计和 MatterGen 生成闭环。
- 完整 Python 源码、4 组回归测试、MatterGen 适配器和便携模板策略。
- evaluator 实际读取的 1000 条结构及其聚合结果。
- 15 条通过验证的原理、45 组关键对照实验、1000 条逐项 `E_hull` / S.U.N. 数据。
- 架构、结果、复现和代码数据文档。

## 结果摘要

- in-sample strict S.U.N.: `344/1000 = 0.344`。
- `E_hull < 0.03`: `592/1000`；`E_hull < 0.10`: `884/1000`。
- 1000 个不同约化化学式，evaluator 报告 100% 成分与结构新颖性。
- evaluator 几何口径下仅 37 个结构有效，其中 10 个同时满足 `E_hull < 0`。

## 已知限制

- 当前结果只完成 in-sample evaluator 评估，尚无 out-of-sample、DFT 或实验确认。
- 最终化学空间高度集中于 Br/Rb 体系。
- evaluator 的 `structural_validity` 与 oracle `validity_rate` 口径不同，不能只报告后者。
- 冻结 `input.json` 缺少 `sites[*].properties` 空字典；Pymatgen 与本次 evaluator 可读取，但严格 schema 消费者可能要求补齐。
- 完整动态轨迹依赖外部 LLM、MatterGen、checkpoint、评估数据和计算资源，不能只靠本精简包逐字节重放。
