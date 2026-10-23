# APRnet

APRnet 将流程分为三个阶段：

- `teacher`：CAnYA 共享监督预训练、CPAD 短肽微调和短肽推理；
- `encoder`：多尺度 APR 编码，输出残基级 HAPR、pAPR 和 rAPR；
- `protein`：ESM-2 650M、HAPR 与 PDB 结构图的蛋白级分类器。

本仓库只发布代码、必要的源数据说明和两个保留的预训练权重。随机划分、同源划分、冷启动划分、1:1–1:5 cohort 划分及其 fold 文件均为外部生成产物，不放入仓库。请使用以下脚本在外部数据目录生成：

- `teacher/my_loader/make_canya_split.py`：CAnYA 随机 90/10 划分；
- `teacher/my_loader/make_cpad_folds.py`：CPAD 短肽随机 K 折；
- `encoder/tools/make_folds.py`：APR 编码器随机或基于外部 group 映射的 K 折；
- `protein/tools/README.md`：蛋白级划分目录契约。

训练命令必须显式传入外部划分路径。ESM-2、HAPR/pAPR/rAPR、PDB/mmCIF、图缓存和实验输出也应放在仓库外部。详细用法见各阶段的 `README.md`。
