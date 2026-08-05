# 复现说明

本项目有三个不同的复现层级，应明确区分。

## 层级 1：核验已发布结果

这一级不调用 LLM、MatterGen 或稳定性模型，只核对冻结结构、逐材料能量和聚合指标是否对齐。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_release.py
```

预期关键输出：1000 个结构、1000 个不同约化化学式、344 个 strict in-sample S.U.N.、稳定性四个互斥区间为 `344/248/292/116`，evaluator 几何口径有效结构为 37。

## 层级 2：运行公开模板基线

```bash
./generate.sh
```

默认设置：

- 输出 `generated/input.json`，不覆盖冻结 `input.json`；
- `TARGET_COUNT=1000`；
- `SEED=20260502`；
- 使用 `strategies/release_template_strategy.json`；
- 关闭指向未公开历史目录的 elite replay。

快速烟测：

```bash
TARGET_COUNT=20 CANDIDATE_MULTIPLIER=20 ./generate.sh
```

模板基线验证代码路径和格式，但不会重建最终 MatterGen 1000 条结果。

默认 1000 条模板生成可能持续占用多个 CPU 核；在共享集群上应提交到计算节点，并通过 `JOBS` 设置并行度，不要在登录节点执行全量生成。

## 层级 3：重新运行完整 agent + MatterGen 闭环

额外需要：

- OpenAI-compatible Responses API 端点和模型；
- MatterGen 源码、兼容 CUDA/PyTorch/PyG 环境与 checkpoint；
- 官方 evaluator、训练参考集和 patched phase-diagram 数据；
- GPU；长任务建议使用 Slurm 或等价调度器。

先配置：

```bash
cp .env.example .env
# 编辑 .env；不要提交该文件
set -a && source .env && set +a
export PYTHONPATH="$PWD/src"
```

A–F 单轮起步示例：

```bash
python -m crystal_llm.run_material_physics_mvp \
  --work-dir runs/af \
  --max-rounds 1 \
  --target-count 10 \
  --evaluator-backend local \
  --mattergen-runner local \
  --mattergen-root "$MATTERGEN_ROOT" \
  --mattergen-model-path "$MATTERGEN_MODEL_PATH"
```

X/Y/Z/W 从已发布精简原理状态起步：

```bash
python -m crystal_llm.run_xy_experience_debate \
  --state artifacts/af_principles_state.json \
  --work-dir runs/xyzw \
  --mode experience_xy \
  --generation-protocol sequential_single \
  --candidate-source generator \
  --generator-backend mattergen \
  --mattergen-adapter mattergen_backend_prototype/mattergen_adapter.py \
  --mattergen-root "$MATTERGEN_ROOT" \
  --mattergen-model-path "$MATTERGEN_MODEL_PATH" \
  --mattergen-runner local \
  --evaluator-backend local
```

控制器还需能找到 evaluator 脚本和数据；具体参数以 `--help` 为准。集群运行前必须按现场资源修改 partition/GRES/CPU，不能照搬内部集群配置。

## 独立 evaluator

本包不再分发 evaluator 的大数据或第三方代码。合法取得后，可把目录放在 `external/evaluator/` 或设置 `EVALUATOR_DIR`：

```bash
EVALUATOR_DIR=/path/to/evaluator DEVICE=cuda ./evaluate_full.sh
```

CPU 模式可用于小规模检查，1000 条稳定性评估建议使用 GPU/调度器。

## 测试环境

发布校验环境：Python 3.12.2、NumPy 1.26.4、Pymatgen 2025.6.14、PyTorch 2.7.1、CHGNet 0.4.2、Pandas 2.2.2、pytest 7.4.4。`requirements.txt` 使用较宽的兼容下限；若要做论文级归档，建议另生成锁定环境文件。

## 不能逐字节重放的部分

- 原始多智能体对话、失败重试和数百轮运行目录未包含。
- `artifacts/final_submission_strategy.provenance.json` 的 elite replay 路径指向内部历史目录，仅用于解释来源。
- 生成式模型、LLM 服务和 GPU 数值环境可能引入非确定性。
- 最终 `input.json` 是冻结证据；完整闭环复跑应被视为新的实验，而非保证产生相同 1000 条结构。
