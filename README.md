# DINO-Med3D: Bridging Dimension and Domain Gaps in Volumetric Segmentation via Progressive Adaptation

**English** | [简体中文](#dinomed3d中文说明)

> **This work has been accepted at MICCAI 2026.**
>
> 📄 **Paper (arXiv):** _link coming soon_ — `https://arxiv.org/abs/XXXX.XXXXX`
>
> If you find this code useful, please consider citing our paper (see [Citation](#citation)).

DINOMed3D adapts the 2D self-supervised **DINOv3** Vision Transformer to **volumetric (3D) medical
image segmentation** through a two-stage pipeline:

1. **Stage 1 (2.5D pre-adaptation).** The DINOv3 backbone is fine-tuned on 2.5D slices (the
   previous / current / next slice stacked as the three input channels) with a lightweight
   segmentation head, transferring the ImageNet-scale representation to the medical domain.
2. **Stage 2 (3D segmentation).** The adapted DINOv3 encoder (applied slice-wise) is fused with a
   high-resolution 3D CNN encoder via a gated fusion module, lifted to 3D feature pyramids by a 3D
   adapter, and decoded by a UPerNet-style 3D decoder with a high-resolution refinement head. The
   backbone can be efficiently tuned with **LoRA** or kept frozen.

Evaluation reports per-patient **Dice** and **HD95**.

---

## Table of Contents
- [Code structure](#code-structure)
- [Installation](#installation)
- [Pretrained weights & DINOv3 repo](#pretrained-weights--dinov3-repo)
- [Data structure](#data-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs & checkpoints](#outputs--checkpoints)
- [Citation](#citation)

---

## Code structure

```
dinomed3d/
├── run.py                  # All-in-one entry: toggle Stage1 / Stage2 / Test via flags at top
├── train.py                # CLI training entry (--model, --dataset, --stage1, --stage2, --debug)
├── test.py                 # CLI inference entry (--weights, --stage)
├── requirements.txt
│
├── configs/
│   └── base.py             # Config: model map, paths, hyper-parameters, LoRA settings
│
├── datasets/
│   ├── stage1_dataset.py   # 2.5D slice-level dataset
│   ├── stage2_dataset.py   # 3D volume / clip dataset + collate_fn_3d
│   ├── transforms.py       # 2D and 3D augmentation + normalization + label remapping
│   └── utils.py            # Label scanning + dataset mean/std statistics
│
├── models/
│   ├── backbone.py         # DINOv3 ViT wrapper exposing multi-scale patch features
│   ├── stage1_model.py     # Stage 1: DINOv3 + linear head (2.5D)
│   └── stage2_model.py     # Stage 2: 2D/3D encoders, gated fusion, 3D adapter, UPerNet3D decoder
│
├── losses/
│   └── unified_loss.py     # Combined Cross-Entropy + Dice loss
│
├── engine/
│   ├── trainer.py          # Training / validation loop, checkpointing, progress logging
│   └── evaluator.py        # Inference, Dice/HD95 metrics, prediction saving
│
└── utils/
    ├── misc.py             # Seeding, 2.5D expansion, one-hot helpers
    ├── metrics.py          # Dice (numpy) and HD95 (medpy)
    └── inference.py        # Sliding-window 3D inference
```

---

## Installation

```bash
# (recommended) create a fresh environment, e.g.
conda create -n dinomed3d python=3.10 -y
conda activate dinomed3d

# install dependencies
pip install -r requirements.txt
```

> A recent PyTorch (≥ 2.5.1) with CUDA is required for the DINOv3 backbones.
> `medpy` (HD95) and `peft` (LoRA) are optional — the code degrades gracefully if they are absent.
> If you do not use Weights & Biases, run with `WANDB_MODE=offline` (see [Usage](#usage)).

---

## Pretrained weights & DINOv3 repo

DINOMed3D builds on the official **DINOv3** ViT backbones, loaded locally via `torch.hub`.

1. **Clone the official DINOv3 repository** (used by `torch.hub.load(..., source='local')`):

   ```bash
   git clone https://github.com/facebookresearch/dinov3.git
   ```

   Point `repo_dir` to it (default `./dinov3`, or set `DINOV3_REPO_DIR`).

2. **Download the DINOv3 pretrained checkpoints** and place them in a single folder
   (default `./pretrained`, or set `DINOV3_WEIGHTS_DIR`):

   | Model size | Backbone        | Expected filename                                   | embed_dim |
   |------------|-----------------|-----------------------------------------------------|-----------|
   | `small`    | `dinov3_vits16` | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`      | 384       |
   | `base`     | `dinov3_vitb16` | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`      | 768       |
   | `large`    | `dinov3_vitl16` | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`      | 1024      |

   These weights are released by Meta AI under the DINOv3 license; please obtain them from the
   official source and comply with its terms.

Resulting layout (paths are configurable, these are just the defaults):

```
pretrained/
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
└── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
dinov3/                      # cloned official DINOv3 repo
```

---

## Data structure

Each dataset is organized into `train` / `val` / `test` splits. Every patient is a folder
containing a `ct/` directory of 2D grayscale slices and a matching `label/` directory of mask
slices with **identical filenames**. Masks are PNGs whose integer pixel values encode the classes
(background + foreground labels); the label set is scanned automatically and remapped to contiguous
indices.

```
data/Colon/                      # = dataset_root  (set via DATASET_ROOT or configs/base.py)
├── train/
│   ├── <patient_id>/
│   │   ├── ct/                  # input slices: 000.png, 001.png, ...
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   └── label/               # masks with the SAME filenames as ct/
│   │       ├── 000.png
│   │       └── ...
│   └── ...
├── val/
│   └── <patient_id>/{ct,label}/*.png
└── test/
    └── <patient_id>/{ct,label}/*.png
```

Notes:
- Slices within a patient are ordered with natural sorting, so use zero-padded indices
  (`000.png`, `001.png`, …) or consistent numeric names.
- `.png` and `.jpg` are both supported.
- Dataset mean/std are computed on the training split; per-pixel label remapping is handled
  automatically.

---

## Usage

### Option A — single entry point (`run.py`)

Edit the flags at the top of [run.py](run.py) and run it:

```python
RUN_STAGE1 = True      # run Stage 1 training
RUN_STAGE2 = True      # run Stage 2 training
RUN_TEST   = True      # run inference on the test split
MODEL      = "base"    # "small" | "base" | "large"
DATASET    = None      # None = use configs/base.py default
DEBUG      = False     # True = fast 2-epoch / 2-batch sanity run
```

```bash
# (optional) point to your data / weights without editing configs/base.py
export DATASET_ROOT=/path/to/data/Colon
export DINOV3_REPO_DIR=/path/to/dinov3
export DINOV3_WEIGHTS_DIR=/path/to/pretrained
export WANDB_MODE=offline          # if you do not use W&B

python run.py
```

### Option B — CLI scripts

```bash
# Stage 1 only
python train.py --model base --stage1

# Stage 2 only (loads work_dirs/<...>/stage1_best.pth automatically)
python train.py --model base --stage2

# Both stages
python train.py --model base --stage1 --stage2

# Override the dataset root / quick debug run
python train.py --model base --stage1 --stage2 --dataset /path/to/data/Colon --debug
```

Inference / evaluation:

```bash
# Stage 2 model
python test.py --weights work_dirs/base_Colon/stage2_best.pth --stage 2

# Stage 1 model
python test.py --weights work_dirs/base_Colon/stage1_best.pth --stage 1
```

`test.py` restores `num_classes`, the label mapping, normalization stats, and the model size
directly from the checkpoint.

---

## Configuration

All hyper-parameters live in [configs/base.py](configs/base.py). Key fields:

| Field                         | Default            | Description                                        |
|-------------------------------|--------------------|----------------------------------------------------|
| `img_size`                    | `512`              | Input resolution                                   |
| `model_type`                  | `base`             | Backbone size (`small`/`base`/`large`)             |
| `s1_lr` / `s2_lr`             | `1e-5` / `1e-4`    | Learning rates for Stage 1 / Stage 2               |
| `s1_epochs` / `s2_epochs`     | `50` / `50`        | Max epochs per stage (early-stopping via plateau)  |
| `s2_clip_depth`               | `16`               | Number of slices per 3D training clip              |
| `s2_infer_window`             | `16`               | Sliding-window depth at inference                  |
| `use_lora`                    | auto (`peft`)      | LoRA-tune the backbone in Stage 2 if `peft` exists |
| `lora_r` / `lora_alpha`       | `16` / `16`        | LoRA rank / scaling                                |
| `use_amp`                     | `False`            | Mixed-precision training                           |
| `voxel_spacing`               | `(1.0, 1.0, 1.0)`  | Voxel spacing used for HD95                         |

Paths (`repo_dir`, `dataset_root`, backbone `checkpoint`) default to relative folders and can be
overridden by the environment variables `DINOV3_REPO_DIR`, `DATASET_ROOT`, and
`DINOV3_WEIGHTS_DIR`.

---

## Outputs & checkpoints

Each run writes to `work_dirs/<model_type>_<dataset_name>/`:

```
work_dirs/base_Colon/
├── stage1_best.pth                 # best Stage 1 checkpoint
├── stage2_best.pth                 # best Stage 2 checkpoint
├── progress.log                    # latest-epoch training summary
├── test_metrics.csv                # per-patient Dice / HD95
└── predict_test_stage2/            # predicted mask slices per patient
    └── <patient_id>/pred_000.png ...
```

Checkpoints store the model weights together with `norm_stats`, `num_classes`, and
`label_mapping`, so evaluation is fully self-contained.

---

## Citation

```bibtex
@inproceedings{dinomed3d2026,
  title     = {DINOMed3D: A DINOv3-Driven Two-Stage Framework for 3D Medical Image Segmentation},
  author    = {<Authors>},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

> The arXiv preprint and full citation will be updated here upon release.

---
---

# DINOMed3D

[English](#dinomed3d-a-dinov3-driven-two-stage-framework-for-3d-medical-image-segmentation) | **简体中文**

> **本工作已被 MICCAI 2026 录用。**
>
> 📄 **论文（arXiv）：** _链接即将公布_ —— `https://arxiv.org/abs/XXXX.XXXXX`
>
> 如果本代码对您的研究有帮助，欢迎引用我们的论文（见[引用](#引用)）。

DINOMed3D 通过一个两阶段流程，将二维自监督 **DINOv3** Vision Transformer 适配到**三维（体数据）医学图像分割**任务：

1. **第一阶段（2.5D 预适配）。** 在 2.5D 切片（将前一张 / 当前 / 后一张切片堆叠为三个输入通道）上微调
   DINOv3 主干，并配合一个轻量分割头，将 ImageNet 规模的表征迁移到医学领域。
2. **第二阶段（3D 分割）。** 将适配后的 DINOv3 编码器（逐切片应用）与一个高分辨率三维 CNN 编码器通过门控
   融合模块相结合，再由三维适配器提升为三维特征金字塔，最后由 UPerNet 风格的三维解码器配合高分辨率细化头进行
   解码。主干可使用 **LoRA** 高效微调，也可保持冻结。

评估指标为逐病例的 **Dice** 与 **HD95**。

---

## 目录
- [代码结构](#代码结构)
- [安装](#安装)
- [预训练权重与 DINOv3 仓库](#预训练权重与-dinov3-仓库)
- [数据结构](#数据结构)
- [使用方法](#使用方法)
- [配置](#配置)
- [输出与权重](#输出与权重)
- [引用](#引用)

---

## 代码结构

```
dinomed3d/
├── run.py                  # 一体化入口：在文件顶部用开关控制 Stage1 / Stage2 / Test
├── train.py                # 命令行训练入口（--model, --dataset, --stage1, --stage2, --debug）
├── test.py                 # 命令行推理入口（--weights, --stage）
├── requirements.txt
│
├── configs/
│   └── base.py             # 配置：模型映射、路径、超参数、LoRA 设置
│
├── datasets/
│   ├── stage1_dataset.py   # 2.5D 切片级数据集
│   ├── stage2_dataset.py   # 3D 体数据 / 片段数据集 + collate_fn_3d
│   ├── transforms.py       # 2D 与 3D 数据增强 + 归一化 + 标签重映射
│   └── utils.py            # 标签扫描 + 数据集均值/方差统计
│
├── models/
│   ├── backbone.py         # DINOv3 ViT 封装，输出多尺度 patch 特征
│   ├── stage1_model.py     # 第一阶段：DINOv3 + 线性头（2.5D）
│   └── stage2_model.py     # 第二阶段：2D/3D 编码器、门控融合、3D 适配器、UPerNet3D 解码器
│
├── losses/
│   └── unified_loss.py     # 交叉熵 + Dice 组合损失
│
├── engine/
│   ├── trainer.py          # 训练 / 验证循环、权重保存、进度日志
│   └── evaluator.py        # 推理、Dice/HD95 指标、预测结果保存
│
└── utils/
    ├── misc.py             # 随机种子、2.5D 扩展、one-hot 辅助函数
    ├── metrics.py          # Dice（numpy）与 HD95（medpy）
    └── inference.py        # 滑窗三维推理
```

---

## 安装

```bash
# （推荐）新建独立环境，例如
conda create -n dinomed3d python=3.10 -y
conda activate dinomed3d

# 安装依赖
pip install -r requirements.txt
```

> DINOv3 主干需要较新的 PyTorch（建议 ≥ 2.5.1）及 CUDA。
> `medpy`（HD95）与 `peft`（LoRA）为可选依赖——缺失时代码会自动降级运行。
> 若不使用 Weights & Biases，请设置 `WANDB_MODE=offline`（见[使用方法](#使用方法)）。

---

## 预训练权重与 DINOv3 仓库

DINOMed3D 基于官方 **DINOv3** ViT 主干，通过本地 `torch.hub` 加载。

1. **克隆官方 DINOv3 仓库**（供 `torch.hub.load(..., source='local')` 使用）：

   ```bash
   git clone https://github.com/facebookresearch/dinov3.git
   ```

   将 `repo_dir` 指向该目录（默认 `./dinov3`，或设置环境变量 `DINOV3_REPO_DIR`）。

2. **下载 DINOv3 的预训练权重**，并放入同一文件夹（默认 `./pretrained`，或设置环境变量
   `DINOV3_WEIGHTS_DIR`）：

   | 模型规模 | 主干            | 预期文件名                                          | embed_dim |
   |---------|-----------------|-----------------------------------------------------|-----------|
   | `small` | `dinov3_vits16` | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`      | 384       |
   | `base`  | `dinov3_vitb16` | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`      | 768       |
   | `large` | `dinov3_vitl16` | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`      | 1024      |

   上述权重由 Meta AI 依据 DINOv3 许可协议发布，请从官方渠道获取并遵守其使用条款。

最终目录结构（路径均可配置，以下仅为默认值）：

```
pretrained/
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
└── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
dinov3/                      # 克隆的官方 DINOv3 仓库
```

---

## 数据结构

每个数据集划分为 `train` / `val` / `test` 三个子集。每个病例为一个文件夹，内含一个 `ct/` 目录（存放二维灰度
切片）和一个对应的 `label/` 目录（存放掩膜切片），二者**文件名必须完全一致**。掩膜为 PNG 图像，其整数像素值即
类别编码（背景 + 前景标签）；标签集合会被自动扫描并重映射为连续索引。

```
data/Colon/                      # = dataset_root（通过 DATASET_ROOT 或 configs/base.py 设置）
├── train/
│   ├── <patient_id>/
│   │   ├── ct/                  # 输入切片：000.png, 001.png, ...
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   └── label/               # 掩膜，文件名与 ct/ 完全相同
│   │       ├── 000.png
│   │       └── ...
│   └── ...
├── val/
│   └── <patient_id>/{ct,label}/*.png
└── test/
    └── <patient_id>/{ct,label}/*.png
```

注意事项：
- 同一病例内的切片按自然排序排列，请使用补零索引（`000.png`、`001.png` ……）或一致的数字命名。
- 同时支持 `.png` 与 `.jpg`。
- 数据集均值/方差在训练集上计算；逐像素标签重映射会自动完成。

---

## 使用方法

### 方式一 —— 一体化入口（`run.py`）

编辑 [run.py](run.py) 顶部的开关后直接运行：

```python
RUN_STAGE1 = True      # 运行第一阶段训练
RUN_STAGE2 = True      # 运行第二阶段训练
RUN_TEST   = True      # 在 test 集上运行推理
MODEL      = "base"    # "small" | "base" | "large"
DATASET    = None      # None = 使用 configs/base.py 默认值
DEBUG      = False     # True = 快速 2 epoch / 2 batch 健全性测试
```

```bash
# （可选）无需修改 configs/base.py，直接指定数据 / 权重路径
export DATASET_ROOT=/path/to/data/Colon
export DINOV3_REPO_DIR=/path/to/dinov3
export DINOV3_WEIGHTS_DIR=/path/to/pretrained
export WANDB_MODE=offline          # 若不使用 W&B

python run.py
```

### 方式二 —— 命令行脚本

```bash
# 仅第一阶段
python train.py --model base --stage1

# 仅第二阶段（自动加载 work_dirs/<...>/stage1_best.pth）
python train.py --model base --stage2

# 两个阶段
python train.py --model base --stage1 --stage2

# 覆盖数据根目录 / 快速调试
python train.py --model base --stage1 --stage2 --dataset /path/to/data/Colon --debug
```

推理 / 评估：

```bash
# 第二阶段模型
python test.py --weights work_dirs/base_Colon/stage2_best.pth --stage 2

# 第一阶段模型
python test.py --weights work_dirs/base_Colon/stage1_best.pth --stage 1
```

`test.py` 会直接从权重文件中恢复 `num_classes`、标签映射、归一化统计量以及模型规模。

---

## 配置

所有超参数都在 [configs/base.py](configs/base.py) 中。关键字段：

| 字段                          | 默认值              | 说明                                       |
|-------------------------------|--------------------|--------------------------------------------|
| `img_size`                    | `512`              | 输入分辨率                                  |
| `model_type`                  | `base`             | 主干规模（`small`/`base`/`large`）          |
| `s1_lr` / `s2_lr`             | `1e-5` / `1e-4`    | 第一 / 第二阶段学习率                        |
| `s1_epochs` / `s2_epochs`     | `50` / `50`        | 各阶段最大 epoch（按 plateau 提前停止）      |
| `s2_clip_depth`               | `16`               | 每个三维训练片段的切片数                     |
| `s2_infer_window`             | `16`               | 推理时滑窗的深度                            |
| `use_lora`                    | 自动（依赖 `peft`） | 若安装了 `peft`，第二阶段对主干使用 LoRA     |
| `lora_r` / `lora_alpha`       | `16` / `16`        | LoRA 秩 / 缩放系数                          |
| `use_amp`                     | `False`            | 混合精度训练                               |
| `voxel_spacing`               | `(1.0, 1.0, 1.0)`  | 计算 HD95 所用的体素间距                     |

路径（`repo_dir`、`dataset_root`、主干 `checkpoint`）默认指向相对目录，可通过环境变量
`DINOV3_REPO_DIR`、`DATASET_ROOT`、`DINOV3_WEIGHTS_DIR` 覆盖。

---

## 输出与权重

每次运行的结果写入 `work_dirs/<model_type>_<dataset_name>/`：

```
work_dirs/base_Colon/
├── stage1_best.pth                 # 第一阶段最佳权重
├── stage2_best.pth                 # 第二阶段最佳权重
├── progress.log                    # 最新 epoch 的训练摘要
├── test_metrics.csv                # 逐病例 Dice / HD95
└── predict_test_stage2/            # 每个病例的预测掩膜切片
    └── <patient_id>/pred_000.png ...
```

权重文件同时保存了模型参数以及 `norm_stats`、`num_classes` 和 `label_mapping`，因此评估过程完全自包含。

---

## 引用

```bibtex
@inproceedings{dinomed3d2026,
  title     = {DINOMed3D: A DINOv3-Driven Two-Stage Framework for 3D Medical Image Segmentation},
  author    = {<Authors>},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

> arXiv 预印本与完整引用信息将在发布后于此更新。
