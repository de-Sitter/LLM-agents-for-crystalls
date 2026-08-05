# GitHub 发布前检查清单

以下项目中，前四项应在公开推送前由项目所有者完成。

## 必须确认

- [ ] 填写仓库名称、作者、机构、联系方式和项目主页。
- [ ] 选择并添加正式 `LICENSE`。当前没有许可证；不应由工具代替权利人决定。
- [ ] 填写 `CITATION.cff.template`，重命名为 `CITATION.cff`。
- [ ] 确认课程 PDF、evaluator 代码/数据、训练参考集和 MatterGen 权重的再分发权；当前均未打包。
- [ ] 明确 README 中所有数值是 in-sample evaluator 结果，不能写成 out-of-sample、DFT 或实验验证。
- [ ] 决定是否保留 evaluator 原样读取的 `input.json`，或另发布补齐 `sites[*].properties={}` 的 schema-normalized 副本；两者需分别记录校验和。

## 本地校验

```bash
python scripts/verify_release.py
PYTHONPATH=src pytest -q
python -m compileall -q src tests mattergen_backend_prototype scripts
rg -n '/data/home/|api\.budprimordium|api\.ikuncode|sk-[A-Za-z0-9]' -g '!docs/PRE_RELEASE_CHECKLIST.md' .
find . -type f -size +50M -print
```

`rg` 与大文件检查应无输出。随后核对 `CHECKSUMS.sha256`。

如果所有发布文件已经定稿，可重建校验和后再核验一次：

```bash
python scripts/verify_release.py --write-checksums
python scripts/verify_release.py
```

## 初始化并推送

本文件夹尚未初始化 Git，也没有外部仓库地址。确认以上事项后再执行：

```bash
cd github_release
git init
git branch -M main
git add .
git status
git commit -m "Initial public release"
git remote add origin git@github.com:OWNER/REPOSITORY.git
git push -u origin main
```

如果使用 GitHub CLI，也可在审阅 `git status` 后执行：

```bash
gh repo create OWNER/REPOSITORY --public --source=. --remote=origin --push
```

## 建议的 GitHub 仓库设置

- About：`Multi-agent crystal structure discovery with validated materials principles, MatterGen generation, and evaluator feedback.`
- Topics：`materials-science`, `crystal-structure`, `multi-agent`, `llm`, `mattergen`, `pymatgen`。
- 开启 secret scanning、Dependabot 和 branch protection。
- 首个 Release 附带 `input.json`、`results.json`、`artifacts/final_materials.csv` 与校验和。
- 若发布论文或正式报告，把大体积不可变数据归档到 DOI 数据仓库，并在 Release 中记录 DOI。
