# 代码与数据

## 仓库内容

| 路径 | 用途 |
|---|---|
| `src/crystal_llm/*.py` | 完整、可导入的最终核心代码；避免只发布入口却缺依赖模块 |
| `tests/test_*.py` | LLM 客户端、A–F、X/Y/Z/W、MatterGen 适配器回归测试 |
| `mattergen_backend_prototype/` | MatterGen 请求 schema、适配器、示例与 LLM 契约 |
| `strategies/release_template_strategy.json` | 去除内部 replay 依赖的便携模板基线 |
| `input.json` | evaluator 实际读取的 1000 条结构快照 |
| `results.json` | 与 `input.json` 对齐的聚合评估 |
| `artifacts/final_materials.csv` | 1000 条逐项能量与 S.U.N. 标签 |
| `artifacts/intermediate_data.md` | 15 条原理、45 组对照实验和最终材料表 |
| `artifacts/af_principles_state.json` | X/Y 可读取的精简原理书状态 |
| `artifacts/dynamic_cumulative_report.json` | 452+548 合并来源索引 |
| `artifacts/final_submission_strategy.provenance.json` | 内部最终策略的只读 provenance 快照 |
| `SOURCE_MANIFEST.json` | 每个抽取文件的来源和 SHA-256 |
| `LICENSE` | 项目自有代码采用的 Apache License 2.0 全文 |

## 外部依赖

| 内容 | 获取与配置 |
|---|---|
| evaluator 参考数据和第三方代码 | 按 [复现说明](REPRODUCIBILITY.md) 配置 `EVALUATOR_DIR` |
| patched phase-diagram 数据（约 2 GB） | 从合法数据源取得后放入 evaluator 环境 |
| MatterGen checkpoint 与独立环境 | 按上游 MatterGen 说明安装，并配置相应环境变量 |
| LLM 服务凭据 | 复制 `.env.example` 为本地 `.env` 并填写自己的服务配置；不要提交密钥 |

## 文件完整性

`SOURCE_MANIFEST.json` 记录核心源码与数据快照的来源和 SHA-256；根目录 `CHECKSUMS.sha256` 用于核对仓库文件完整性。

## 许可证

根目录 `LICENSE` 适用于本项目自有代码。外部依赖、MatterGen、checkpoint、evaluator 数据及其他第三方材料不因包含或引用于本仓库而重新授权，使用时须分别遵守其原许可证和服务条款。
