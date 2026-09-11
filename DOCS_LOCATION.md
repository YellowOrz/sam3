# 文档与代码分开存放

2026-09-11 按用户要求，将报告、图片和复核资料从代码仓库移至同级目录。

服务器实际目录：

- 代码：`/home/zhengyuxi/projects/sam3-yelloworz/`
- 文档：`/home/zhengyuxi/projects/docs/`

Windows 对应建议目录：

- 代码：`C:\Users\jixiegeming\Desktop\paper_review\sam3-yelloworz\`
- 文档：`C:\Users\jixiegeming\Desktop\paper_review\docs\`

服务器仓库内的 `docs -> ../docs` 是仅供本地兼容的符号链接，已被 Git 忽略；它不是第二份文档，也不会推送到 GitHub。全新 clone 不会自动带上外部文档或复核数据。

迁移前对文档内容、Git 暂存区补丁和 index 作了备份，位置为服务器仓库下的 `runs/docs-relocation-20260911-7aORBP/`；`docs-before.tar.gz` 已逐项与迁移前目录比较通过。只撤销 docs 的暂存，不丢弃其他已暂存代码，不改写原始审计记录或封存 ZIP。

## 当前阅读入口

- [输出残差完整讲解](../docs/sam3-output-residual-explained.md)
- [输出残差示意图 PNG](../docs/figures/sam3-output-residual.png)
- [夜间实验报告](../docs/overnight-experiment-results-2026-09-11.md)
- [SAM3 loss 说明](../docs/sam3-loss-reference.md)

这些相对链接面向本机的同级目录布局，在 GitHub 网页中不代表已发布的文档。历史文档包含服务器绝对路径、旧相对源码链接和源码指纹，迁移不批量重写这些历史证据；需要便携阅读时使用显式文档根目录生成的新分类包。

文档、训练数据、checkpoint、运行输出与凭据不随代码提交。代码仓库仍保留 CHANGELOG，用来追踪实验和实现改动。
