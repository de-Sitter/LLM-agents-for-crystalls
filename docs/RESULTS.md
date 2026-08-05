# 结果与解释边界

## 最终 1000 条结果

| 区间 | 数量 | 比例 |
|---|---:|---:|
| `E_hull < 0` | 344 | 34.4% |
| `0 <= E_hull < 0.03` | 248 | 24.8% |
| `0.03 <= E_hull < 0.10` | 292 | 29.2% |
| `E_hull >= 0.10` | 116 | 11.6% |

`E_hull` 最小值、均值、中位数分别为 `-0.262347`、`0.024828`、`0.022119 eV/atom`。所有逐材料数值位于 `artifacts/final_materials.csv`；聚合原始输出位于 `results.json`。

## 有效性、唯一性与新颖性

- 约化化学式唯一：1000/1000。
- evaluator 结构唯一：1000/1000。
- in-sample 成分新颖与结构新颖：均为 100%。
- evaluator 成分有效：1000/1000。
- evaluator 几何结构有效：37/1000；其中 10 个同时满足 `E_hull < 0`。

`results.json` 中 `success_rate.validity_rate=1.0` 来自 oracle evaluation 的 `MaterialsEvaluation.valid`；`validity.structural_validity=0.037` 来自另一套几何规则。公开汇报必须同时保留这两个口径，不能把前者解释成 100% 几何有效。

## 几何口径内的严格稳定候选

下表仅按当前 evaluator 的 `volume/atom < 30 Å³` 几何条件和预测 `E_hull` 排序，仍不是 DFT 或实验确认：

| 输入序号 | 化学式 | `E_hull` (eV/atom) |
|---:|---|---:|
| 327 | LiVBr6 | -0.225357119 |
| 337 | Li2VBr6 | -0.130726027 |
| 336 | Li(VBr4)2 | -0.130436309 |
| 329 | Li3VBr8 | -0.110036503 |
| 334 | Li3(VBr5)2 | -0.089832930 |
| 185 | Rb2Ni3Br10 | -0.041371298 |
| 314 | Na(VBr3)2 | -0.038066405 |
| 335 | Li(VBr3)2 | -0.036704568 |
| 328 | LiVBr4 | -0.029534816 |
| 323 | NaV3Br8 | -0.015967046 |

## 化学空间分布

- 942/1000 含 Br，882/1000 含 Rb，280/1000 含 Cd。
- 528 个三元、466 个四元、6 个五元结构。
- 结果高度集中在碱金属卤化物，尤其是 Rb/Br 子空间；高唯一性不等同于元素空间均匀覆盖。

这个集中性说明闭环找到了高产稳定盆地，但也意味着 34.4% S.U.N. 不能外推为任意化学空间的平均生成能力。

## A–F 原理成果

`artifacts/intermediate_data.md` 包含：

- 15 条主状态中验证通过的材料学原理；
- 45 组关键 primary/control 对照及每个材料的最终 `E_hull`；
- 1000 个最终材料的化学式、能量与 S.U.N. 标签。

原理书特意保留边界，例如 Mg/Ca 的 F 掺杂迁移差异、P–S 机制不能直接迁移到 P–O、Mg–Si–N 的失败边界、Ba 尺寸边界，以及 Rb–Cd–Br 不能扩展成 Cd>Zn 或 Rb>Cs 的普遍排序。

## 必须保留的声明

1. 当前是 in-sample evaluator 结果；课题最终排名所需 out-of-sample 结果未包含。
2. `E_hull` 来自 CHGNet + patched phase diagram 流程，不是重新执行的 DFT。
3. 结构尚未经过实验合成、声子、有限温度、动力学稳定性或完整 DFT 弛豫确认。
4. 负 `E_hull` 应解释为该 evaluator 中的候选优先级信号，不应直接写成“已发现实验稳定新材料”。
5. 冻结 JSON 可被 Pymatgen 与当前 evaluator 读取，但 `sites[*].properties` 缺失，严格 schema 使用者需先补空字典并重新记录校验和。

