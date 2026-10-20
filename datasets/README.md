# 数据集

`EORSSD/` 为现有数据集，`ORSSD/` 包含 600 个训练样本和 200 个测试样本。

ORSSD 按 EORSSD 的子集编号组织，训练和测试划分保持源数据不变：

- `train-images`、`train-labels`、`test-images`、`test-labels`：来自 `F:\DataSets\ORSSD`。
- `train-nam-npz`、`test-nam-npz`：使用 ORSSD 自身的 NAM 分区及清单。
- `train-mean/region_mean_60`、`test-mean/region_mean_60`：来自 `D:\Visual Studio Projects\NAMLab\x64\ORSSD`，保留 60 层区域均值图结构。
- `train-scribble-gt`、`train-scribble-mask`：来自 `F:\DataSets\ORSSD-S`。
- `train-seor-gt`、`train-seor-mask`：从 `F:\DataSets\S-EOR\train` 按训练编号复制。

已核对：ORSSD 的 600 个训练编号、200 个测试编号均属于 EORSSD 的对应划分。600 张训练标签像素一致；测试标签 `0006`、`0136`、`0746` 存在像素差异。部分同编号图像和 NAM 分区也存在差异，因此保留 ORSSD 自身的图像、标签和 NAM 配套版本。原始数据保留在 `F:\DataSets`。

训练和测试入口仍默认使用 EORSSD；使用 ORSSD 时，通过现有命令行参数指定 `datasets/ORSSD/` 下的对应目录。

EORSSD 已从完整源数据恢复 `train-scribble-mask`、`train-seor-gt`、`train-seor-mask`；来源分别为 `F:\DataSets\EORRSD-S` 和 `F:\DataSets\S-EOR\train`。EORSSD 的 `train-mean`、`test-mean` 来自 `D:\Visual Studio Projects\NAMLab\x64\EORSSD`。

两套 mean 均已核对与各自现有图像、NAM 分区一致。替换前的 7 个目录保存在 `.maintenance/eorssd-repair/backup`（相对于 NGS 项目根目录），验证记录位于同级 `verification.json`。
