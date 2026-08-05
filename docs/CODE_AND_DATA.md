# 代码与数据清单

## 已包含

| 路径 | 发布理由 |
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

## 未包含

| 内容 | 原因 / 替代办法 |
|---|---|
| `.env`、API key、内部 endpoint | 安全；只发布 `.env.example` |
| `archive/.../2024-08-07-ppd-mp.pkl`（约 2 GB） | 超出普通 GitHub 发布范围；由合法数据源单独取得 |
| evaluator 参考数据和第三方代码 | 许可与体积边界；通过 `EVALUATOR_DIR` 外部挂载 |
| MatterGen checkpoint 与独立环境 | 大文件、硬件和上游许可边界 |
| `physics_mvp_runs/`、`xy_runs/` 全量历史 | 体积大、包含瞬态日志；精简为状态、原则表和来源 manifest |
| controller/Slurm/evaluator 原始日志 | 包含机器路径与大量进度输出；结果已结构化保存 |
| stale/before-merge 结果 | 防止用户把旧快照误当最终结果 |
| 课程 PDF | 发布权未确认；本包只总结任务要求 |

## 文件完整性

`SOURCE_MANIFEST.json` 同时记录原工作区源文件和清理后发布副本的 SHA-256。根目录的最终 `CHECKSUMS.sha256`（生成于发布校验阶段）用于核对公开包本身。

发布包没有单个超过 GitHub 100 MB 硬限制的文件；`input.json` 约 2.7 MB，整个包约 6 MB，不需要 Git LFS。若今后加入模型权重、phase diagram 或完整运行历史，应改用 Git LFS、Zenodo/对象存储，并在仓库中只保留可验证 manifest。

## 代码清理范围

公开副本仅做以下非算法性处理：

- 将个人 MatterGen 绝对路径改为环境变量/相对默认值；
- 将内部中转站测试 URL 改为无效示例域名；
- 将测试中的项目绝对路径改为相对测试目录；
- 不复制内部 Slurm launcher 和 evaluator 日志。

抽取逻辑保存在原工作区 `scripts/build_github_release_package.py`，便于后续从 `final_version/` 重建本目录。

## 许可边界

根目录 `LICENSE` 适用于本项目自有代码。外部依赖、MatterGen、checkpoint、evaluator 数据及其他第三方材料不因包含或引用于本仓库而重新授权，使用时须分别遵守其原许可证和服务条款。
