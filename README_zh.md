# APRnet 代码结构

2026-10-20 已修复结构路径、窗口输出均值、缓存序列校验、训练入口和 mean±SD 汇总检查，详见 [修改记录](CHANGELOG.md)。数据表、划分、网络层和两份权重文件均保持原样。

**权重配套仍有待解决**：保存记录显示，epoch20 encoder 使用的 teacher 输入与当前打包的 teacher 不同。现在会明确拒绝不兼容的组合，不再静默混用；没有替换权重或重新训练。新版提取只对真实残基计算窗口概率均值，已加版本标记，不改变 CNN 的 padding。详见 [权重配套说明](encoder/pretrained/README.md)。

只分三个阶段：`teacher`、`encoder`、`protein`。每个阶段都按 PHCNet 的习惯分为 `model`、`my_loader`、`train`、`utils`、`scratch/get_feature`、`feature`、`data` 和 `result`。

- **teacher**：CAnYA 共享监督预训练、CPAD 微调、最终 teacher 训练和短肽预测。`pretrained/teacher_canya_cpad.pt` 是已有最终权重。
- **encoder**：六个尺度的 teacher 窗口输入、残基级编码、训练目标和 HAPR 提取。`pretrained/encoder_epoch20.pt` 是保留的 epoch20 权重。
- **protein**：只保留 **ESM-2 650M＋HAPR＋PDB**。全局 Cross-Attention、pAPR/rAPR 门控与池化、结构图及物理链实例读出均按原模型保留。

下游的纯 PLM、无 PDB、其他 ESM-2 大小、局部 attention、Joint Regions、超参数扫描和服务器队列脚本不再放进这个版本。

**不打包已经生成的 HAPR、pAPR/rAPR、ESM-2 表征和提取缓存。** 提取代码保留；生成文件通过参数写到仓库外部。teacher 的固定氨基酸描述符表属于模型输入定义，保留在 `teacher/feature`。

所有命令从 APRnet 根目录用 `python -m ...` 运行。训练需要 Linux 与匹配 CUDA 的 PyTorch。三个阶段的使用方法分别见各自 `README.md`；本次只重构代码并复制已有权重，不重新训练，不修改原始数据、划分、论文或实验结果。权重的校验值和历史来源说明在各自 `pretrained` 目录。

## 已包含的数据

- `teacher/data`：CAnYA 实际输入表、2,031 条短肽合并 Excel、原始十折数据与划分。
- `encoder/data`：3,031 条合并 Excel（332 条 APR 长序列、668 条背景蛋白、2,031 条 CPAD 短肽）、来源审计和两种十折划分。
- `protein/data`：1,417 条 primary 合并 Excel、primary 与 1:1～1:5 的来源数据表和四种划分，合计 210 个 fold。
- `protein/data/merged_protein_cohorts.xlsx`：便于查看的六套样本总表，包含标签、序列 ID、测试折编号和 6,557 条完整链序列。原训练输入和划分仍以随附的原始文件为准。

原 Excel、TSV、CSV 逐文件复制并校验 SHA-256，不重新筛样本、不修改标签或划分。数据说明、来源表和固定划分保留在各阶段的 `data` 目录。PDB/mmCIF 坐标文件仍作为外部数据，包内保留结构编号、已有来源链接和链序列对齐表。生成的 HAPR、pAPR/rAPR、ESM-2 embedding、结构图及其他提取缓存继续排除。下游分类器不附权重。

## 环境与安装

安装方法、已验证的依赖版本及检查命令统一放在 [README 的安装说明](README.md#installation)，不再单独设置 `docs` 目录。需要单独准备匹配 GPU 的 PyTorch、teacher 使用的 ESM-2 8M、蛋白模型使用的 ESM-2 650M，以及已有对齐信息对应的 PDB/mmCIF 坐标文件。

根目录不再包含代码迁移清单和检查报告；这些整理记录已在仓库外归档。模型配置、固定氨基酸描述符、权重说明和数据划分协议仍保留在对应模块中。
