# 基于多层次先验知识的 Few-Shot 图像分类研究与实现

> A systematic and reproducible study of Few-Shot Image Classification based on multi-level prior knowledge.

本项目围绕 **Few-Shot Image Classification（少样本图像分类）** 展开，基于 Mini-ImageNet 构建统一的实验框架，对比不同类型的先验知识与模型适配策略在少样本分类任务中的表现。

项目最初来源于本科毕业设计，毕业后对原有实验进行了系统性重构与扩展。相比单纯完成一个 Few-shot 分类任务，本项目更关注：

* 不同 Few-shot 方法如何在统一条件下进行公平比较；
* 预训练视觉知识、文本语义知识等不同先验知识的作用；
* Prompt、Prototype、Feature Cache、Adapter、LoRA 等不同适配机制的特点；
* 少样本条件下模型性能与可训练参数量之间的关系。

---

## 1. 实验设置

### Dataset

主要使用 **Mini-ImageNet** 进行实验。

* 类别数：100
* 每类训练样本：20 张
* 任务：100-way image classification
* 评价指标：Classification Accuracy

为了研究不同先验知识的作用，项目从随机初始化模型开始，逐步引入 ImageNet 预训练视觉特征以及 CLIP 的视觉-语言先验。

---

## 2. 实验体系

项目按照“**数据 → 视觉预训练 → 原型学习 → 视觉-语言预训练 → 参数/特征适配**”的思路逐步构建实验。

### 2.1 White-Box Baseline

首先使用随机初始化的 **ResNet18**，配合数据增强和 Linear Classifier 建立基础基线。

该实验主要用于观察：

> 在没有任何预训练知识，仅依赖少量目标任务数据的情况下，模型能够达到怎样的性能。

---

### 2.2 ImageNet Pretrained ResNet18

随后引入 ImageNet 预训练权重，并比较不同训练方式：

* Linear Classifier
* Frozen Backbone + Linear Classifier
* Prototype Network

用于分析**预训练视觉特征**以及**原型度量学习**在 Few-shot 场景中的作用。

---

### 2.3 CLIP Zero-shot

进一步引入 **CLIP ViT-B/32**，利用其大规模图文预训练获得的视觉-语言先验。

通过：

```text
Image → Image Encoder → Image Feature
                         ↓
                      Similarity
                         ↑
Text → Text Encoder → Text Feature
```

计算图像特征与类别文本特征之间的相似度，在不使用目标任务训练样本的情况下进行 Zero-shot 分类。

---

### 2.4 CLIP Linear Probe

在 CLIP 预训练特征基础上，仅利用少量目标任务样本训练分类器，研究少量监督数据对预训练视觉特征的进一步适配效果。

---

### 2.5 Knowledge-enhanced Hard Prompt

针对 CLIP 的文本侧表示，引入更加丰富的类别语义描述，通过构建知识增强的文本 Prompt 提升类别文本特征的信息量。

例如从简单的：

```text
a photo of a dog
```

扩展为包含类别属性、外观等更加细粒度信息的文本描述。

---

### 2.6 CoOp

实现 **Context Optimization (CoOp)**，将人工设计的离散 Prompt 转换为可学习的连续 Context，通过训练 Prompt 参数使文本表示更加适应目标分类任务。

---

### 2.7 Tip-Adapter

引入 **Feature Cache**，将少量训练样本对应的视觉特征作为外部特征记忆，并与 CLIP 原有的 Zero-shot 预测结果结合，实现无需大规模更新模型参数的快速适配。

---

### 2.8 CLIP-Adapter

通过轻量级 Adapter 对 CLIP 的视觉/语义特征进行任务相关的调整，在保留预训练知识的同时学习少量目标任务信息。

---

### 2.9 LoRA

实现 **Low-Rank Adaptation (LoRA)**，通过低秩矩阵学习模型参数更新，而不是直接更新完整模型参数。

本项目重点比较 LoRA 在少样本场景下的：

* 分类性能
* 可训练参数量
* 参数效率

---

## 3. 实验结果

在统一的 Mini-ImageNet 100 类、每类 20 张训练图片的设置下，主要实验结果如下：

| 方法                                  | Test Accuracy |
| :---------------------------------- | ------------: |
| Random Init. ResNet18               |    **28.51%** |
| ImageNet Pretrained ResNet18        |    **67.35%** |
| Frozen Backbone + Linear Classifier |    **87.28%** |
| CLIP Zero-shot                      |    **83.75%** |
| CLIP Linear Probe                   |    **88.65%** |
| Hard Prompt                         |    **86.98%** |
| CoOp                                |    **88.93%** |
| Tip-Adapter                         |    **86.67%** |
| LoRA                                |    **89.82%** |

### LoRA 参数效率

LoRA 在本项目实验中取得最高测试准确率：

| 指标                   |          结果 |
| :------------------- | ----------: |
| Test Accuracy        |  **89.82%** |
| Trainable Parameters | **36.86 万** |
| Trainable Ratio      | **0.2431%** |

即仅训练约 **0.24%** 的模型参数，即可获得本实验中最高的测试准确率。

---

## 4. 实验结论

### 4.1 预训练知识是 Few-shot 分类的重要基础

随机初始化 ResNet18 的测试准确率仅为 **28.51%**，加入 ImageNet 预训练后提升至 **67.35%**。

这说明在训练样本极少的情况下，模型已有的视觉知识能够显著缓解目标任务数据不足的问题。

### 4.2 充分利用预训练特征比直接更新整个模型更加稳定

冻结预训练视觉 Backbone 后，仅训练分类器即可达到 **87.28%**。

这表明在极少样本条件下，直接使用少量数据更新大量模型参数可能导致过拟合，而充分利用已经学习到的通用视觉特征是一种有效策略。

### 4.3 CLIP 的视觉-语言先验具有较强的迁移能力

CLIP Zero-shot 在完全不使用目标任务训练样本的情况下达到 **83.75%**，说明大规模图文预训练获得的语义知识可以直接迁移到新的分类任务。

进一步通过 Linear Probe、Prompt Learning 等方法进行适配，可以继续挖掘预训练模型中的知识。

### 4.4 参数高效微调具有较好的少样本适应能力

LoRA 在仅训练 **0.2431%** 模型参数的情况下获得 **89.82%** 的测试准确率，为本实验中表现最好的方法。

这说明在少样本场景中，通过低秩参数更新保留大部分预训练知识，同时学习少量任务相关信息，是一种具有较高参数效率的适配方式。

---

## 5. 项目方法演进

本项目的整体实验逻辑可以概括为：

```text
                    Few-shot Classification
                              │
             ┌────────────────┴────────────────┐
             │                                 │
       无预训练先验                       利用预训练先验
             │                                 │
      Random ResNet18                    ResNet / CLIP
             │                                 │
             │                    ┌────────────┴────────────┐
             │                    │                         │
             │              Visual Prior              Vision-Language Prior
             │                    │                         │
             │             Linear / ProtoNet       Zero-shot / Linear Probe
             │                                              │
             │                           ┌──────────────────┼──────────────────┐
             │                           │                  │                  │
             │                         Prompt          Feature Cache        Adapter
             │                           │                  │                  │
             │                         CoOp            Tip-Adapter       CLIP-Adapter
             │                                              │
             │                                             LoRA
             └──────────────────────────────────────────────┘
```

项目并非单纯比较不同模型的最终准确率，而是希望从**先验知识来源和适配方式**的角度理解 Few-shot Learning。

---

## 6. 项目结构

```text
FewShot/
│
├── README.md
│
├── WhiteBox_baseline.py
│
├── datalevel_baseline.py
├── datalevel_frozen.py
│
├── modellevel_baseline.py
├── modellevel_ablation.py
│
├── clip_baseline.py
│
├── strategy_linear_fine_tuning.py
├── strategy_hard_prompt.py
├── strategy_coop.py
├── strategy_tip_adapter.py
├── strategy_clip_adapter.py
├── strategy_LoRA.py
│
├── generate_descriptions_v1.py
├── generate_descriptions_v2.py
│
├── split_dataset.py
├── download_resnet.py
│
└── ...
```

### 主要代码说明

| 文件                               | 功能                         |
| :------------------------------- | :------------------------- |
| `WhiteBox_baseline.py`           | 随机初始化 ResNet18 基线          |
| `datalevel_baseline.py`          | 数据层面实验                     |
| `datalevel_frozen.py`            | 冻结 Backbone 的数据层面实验        |
| `modellevel_baseline.py`         | ImageNet 预训练模型基线           |
| `modellevel_ablation.py`         | 模型层面消融实验                   |
| `clip_baseline.py`               | CLIP Zero-shot / 基础实验      |
| `strategy_linear_fine_tuning.py` | Linear Probe / Fine-tuning |
| `strategy_hard_prompt.py`        | 知识增强 Hard Prompt           |
| `strategy_coop.py`               | CoOp                       |
| `strategy_tip_adapter.py`        | Tip-Adapter                |
| `strategy_clip_adapter.py`       | CLIP-Adapter               |
| `strategy_LoRA.py`               | LoRA 参数高效微调                |
| `generate_descriptions_v1.py`    | 类别描述生成                     |
| `generate_descriptions_v2.py`    | 类别描述生成方案迭代                 |
| `split_dataset.py`               | 数据集划分                      |

---

## 7. 环境

项目主要使用：

* Python 3.10
* PyTorch
* torchvision
* CUDA
* NumPy
* PIL
* scikit-learn
* tqdm

建议使用 Conda 创建独立环境：

```bash
conda create -n fewshot python=3.10
conda activate fewshot
```

然后根据具体实验脚本安装所需依赖。

---

## 8. 数据集准备

项目主要使用 Mini-ImageNet 数据集。

由于数据集文件较大，本仓库**不直接提供数据集**。

准备数据后，请根据对应实验脚本中的路径配置放置数据。

运行实验前建议检查：

```text
Dataset Path
Model Path
Train / Val / Test Split
Batch Size
Learning Rate
Device
Checkpoint Path
```

不同实验脚本的具体参数可能存在差异，请以代码中的配置为准。

---

## 9. 运行示例

例如运行 CLIP 基础实验：

```bash
python clip_baseline.py
```

运行不同适配策略：

```bash
python strategy_linear_fine_tuning.py

python strategy_hard_prompt.py

python strategy_coop.py

python strategy_tip_adapter.py

python strategy_clip_adapter.py

python strategy_LoRA.py
```

实验结果会根据对应脚本的设置保存至指定目录。

---

## 10. AI 辅助开发

本项目在开发过程中使用 **GPT** 作为 AI 辅助工具，主要用于：

* 根据论文思路生成代码初稿；
* 解释 PyTorch / 深度学习代码；
* 辅助定位运行错误；
* 协助分析实验结果；
* 对部分代码进行局部优化。

AI 并未直接决定最终实验方案。

项目的实验框架、方法选择、评价标准和实验设计由本人确定；AI 主要用于代码初稿、调试辅助和局部优化。所有实验均由本人实际运行，并对代码逻辑和实验结果进行检查与验证。

特别是在复现论文方法时，重点检查：

```text
论文方法
   ↓
理解核心机制
   ↓
AI 辅助代码实现
   ↓
检查 Tensor Shape / Gradient / Parameter
   ↓
实际运行
   ↓
分析实验结果
```

因此，本项目更强调**理解方法并验证实现是否正确**，而不仅仅是让代码能够运行。

---

## 11. 项目总结

本项目通过统一的 Few-shot 分类实验条件，将传统视觉模型、原型学习、视觉-语言预训练以及参数高效微调等方法放在同一实验框架下进行比较。

核心实验结论可以概括为：

> **少样本学习的关键并不只是增加少量训练技巧，而是如何充分利用模型已有的先验知识，并以合适的方式将少量目标任务信息与这些先验知识结合。**

从随机初始化 ResNet18 到 ImageNet 预训练，再到 CLIP 视觉-语言先验，以及 Prompt、Feature Cache、Adapter 和 LoRA 等适配方法，实验结果体现了不同层次先验知识和适配机制在 Few-shot 场景下的作用。

本项目的主要目标不是提出新的 Few-shot 算法，而是通过**统一实验、方法复现、对比分析与工程实现**，建立对现代 Few-shot Learning 方法体系的整体认识，并形成一套可复现、可扩展的实验代码框架。
"""

print(readme)

 
