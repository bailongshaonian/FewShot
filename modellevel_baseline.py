import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm


# ============================================================
# 1. 配置参数
# ============================================================

DATA_DIR = "mini-imagenet"

MODEL_PATH = os.path.join(
    "models",
    "resnet18_pretrained.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100
K_SHOT = 20

BATCH_SIZE = 128
NUM_WORKERS = 4

SEED = 42


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. 数据预处理
# ============================================================

def get_transform():

    # ProtoNet 不进行当前任务上的参数训练，
    # 因此 Support / Validation / Test 都使用确定性的
    # evaluation transform。
    #
    # 这样可以避免额外的数据增强对 Model-level 实验造成干扰。

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return eval_transform


# ============================================================
# 4. 构建 Frozen Pretrained ResNet18
# ============================================================

def build_backbone():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到预训练模型：{MODEL_PATH}\n"
            f"请先运行 download_pretrained.py"
        )

    print("\n正在加载 ImageNet 预训练 ResNet18...")

    # 创建 ResNet18
    backbone = resnet18(weights=None)

    # 加载本地预训练参数
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    backbone.load_state_dict(state_dict)

    # 保存 feature dimension
    feature_dim = backbone.fc.in_features

    # 移除原始 ImageNet 分类头
    backbone.fc = nn.Identity()

    # 冻结整个 backbone
    for param in backbone.parameters():
        param.requires_grad = False

    backbone = backbone.to(DEVICE)
    backbone.eval()

    print(
        f"Embedding dimension: {feature_dim}"
    )

    return backbone, feature_dim


# ============================================================
# 5. 提取 Embedding
# ============================================================

@torch.no_grad()
def extract_embeddings(
    backbone,
    loader
):

    backbone.eval()

    all_embeddings = []
    all_labels = []

    amp_enabled = DEVICE == "cuda"

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE,
            enabled=amp_enabled
        ):

            embeddings = backbone(images)

        # 转成 float32，避免后续 prototype 计算受到 AMP 精度影响
        embeddings = embeddings.float()

        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(
            targets
        )

    embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    return embeddings, labels


# ============================================================
# 6. 计算 Prototype
# ============================================================

def compute_prototypes(
    embeddings,
    labels,
    num_classes
):

    feature_dim = embeddings.shape[1]

    prototypes = torch.zeros(
        num_classes,
        feature_dim,
        dtype=embeddings.dtype
    )

    for class_id in range(num_classes):

        class_embeddings = embeddings[
            labels == class_id
        ]

        if len(class_embeddings) == 0:
            raise ValueError(
                f"类别 {class_id} 在 Support Set 中没有样本。"
            )

        prototypes[class_id] = (
            class_embeddings.mean(dim=0)
        )

    return prototypes


# ============================================================
# 7. ProtoNet 分类
# ============================================================

def predict_with_prototypes(
    embeddings,
    prototypes
):

    # embeddings:
    # [N, D]
    #
    # prototypes:
    # [C, D]
    #
    # 输出：
    # [N, C]

    # 使用平方欧氏距离
    distances = torch.cdist(
        embeddings,
        prototypes,
        p=2
    ).pow(2)

    predictions = distances.argmin(
        dim=1
    )

    return predictions, distances


# ============================================================
# 8. 评估
# ============================================================

def evaluate_protonet(
    embeddings,
    labels,
    prototypes
):

    predictions, distances = predict_with_prototypes(
        embeddings,
        prototypes
    )

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


# ============================================================
# 9. 保存实验结果
# ============================================================

def save_results(
    train_size,
    val_size,
    test_size,
    val_acc,
    test_acc
):

    results_dir = "results"

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    results_file = os.path.join(
        results_dir,
        "results_data.txt"
    )

    with open(
        results_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write("=" * 70 + "\n")
        f.write(
            "Model-level Baseline\n"
        )
        f.write("=" * 70 + "\n")

        # ----------------------------------------------------
        # 实验身份
        # ----------------------------------------------------

        f.write(
            "Experiment: "
            "model_baseline_frozen_pretrained_protonet\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        # ----------------------------------------------------
        # 模型
        # ----------------------------------------------------

        f.write(
            "Model: ImageNet Pretrained ResNet18\n"
        )

        f.write(
            "Backbone: Frozen\n"
        )

        f.write(
            "Embedding Dimension: 512\n"
        )

        f.write(
            "Classifier: Prototypical Network\n"
        )

        f.write(
            "Distance Metric: Squared Euclidean Distance\n"
        )

        f.write(
            f"Pretrained Weights: {MODEL_PATH}\n"
        )

        # ----------------------------------------------------
        # Few-shot 设置
        # ----------------------------------------------------

        f.write(
            f"Num Classes: {NUM_CLASSES}\n"
        )

        f.write(
            f"K-shot: {K_SHOT}\n"
        )

        # ----------------------------------------------------
        # 数据处理
        # ----------------------------------------------------

        f.write(
            "Support Transform: "
            "Resize + CenterCrop + Normalize\n"
        )

        f.write(
            "Query Transform: "
            "Resize + CenterCrop + Normalize\n"
        )

        # ----------------------------------------------------
        # 数据规模
        # ----------------------------------------------------

        f.write(
            f"Support Size: {train_size}\n"
        )

        f.write(
            f"Validation Size: {val_size}\n"
        )

        f.write(
            f"Test Size: {test_size}\n"
        )

        # ----------------------------------------------------
        # 实验结果
        # ----------------------------------------------------

        f.write(
            f"Validation Accuracy: "
            f"{val_acc:.2f}%\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_acc:.2f}%\n"
        )

        # ProtoNet 不进行 epoch 训练
        f.write(
            "Training Epochs: 0\n"
        )

        f.write(
            "Episodic Meta-training: No\n"
        )

        f.write("\n")

    print(
        f"\n实验结果已追加保存至："
        f"{results_file}"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Model-level Few-shot Baseline")
    print("=" * 70)

    print(
        "ImageNet Pretrained ResNet18"
    )

    print(
        "        ↓"
    )

    print(
        "Frozen Backbone"
    )

    print(
        "        ↓"
    )

    print(
        "512-d Embedding"
    )

    print(
        "        ↓"
    )

    print(
        "20-shot Class Prototypes"
    )

    print(
        "        ↓"
    )

    print(
        "Euclidean Distance Classification"
    )

    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Classes      : {NUM_CLASSES}")
    print(f"K-shot       : {K_SHOT}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Seed         : {SEED}")


    # ========================================================
    # 1. 数据
    # ========================================================

    transform = get_transform()

    train_dir = os.path.join(
        DATA_DIR,
        "train"
    )

    val_dir = os.path.join(
        DATA_DIR,
        "val"
    )

    test_dir = os.path.join(
        DATA_DIR,
        "test"
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transform
    )


    # ========================================================
    # 2. 检查数据
    # ========================================================

    print("\n数据集信息：")

    print(
        f"Support images : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test images : "
        f"{len(test_dataset)}"
    )

    print(
        f"Classes : "
        f"{len(train_dataset.classes)}"
    )

    assert (
        train_dataset.classes
        == val_dataset.classes
    ), "Train 与 Val 类别顺序不一致！"

    assert (
        train_dataset.classes
        == test_dataset.classes
    ), "Train 与 Test 类别顺序不一致！"

    assert (
        len(train_dataset.classes)
        == NUM_CLASSES
    ), f"类别数量不是 {NUM_CLASSES}！"

    # 理论上 train 应该是 20-shot
    expected_train_size = (
        NUM_CLASSES * K_SHOT
    )

    if len(train_dataset) != expected_train_size:

        print(
            "\n警告："
            f"Train 数据量为 {len(train_dataset)}，"
            f"而按照 {K_SHOT}-shot × "
            f"{NUM_CLASSES} 类应为 "
            f"{expected_train_size}。"
        )


    # ========================================================
    # 3. DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )


    # ========================================================
    # 4. Backbone
    # ========================================================

    backbone, feature_dim = build_backbone()


    # ========================================================
    # 5. 提取 Support Embeddings
    # ========================================================

    print(
        "\n正在提取 Support Set embeddings..."
    )

    support_embeddings, support_labels = extract_embeddings(
        backbone,
        train_loader
    )

    print(
        f"Support embedding shape: "
        f"{tuple(support_embeddings.shape)}"
    )


    # ========================================================
    # 6. 构建 Prototype
    # ========================================================

    print(
        "\n正在计算类别 prototypes..."
    )

    prototypes = compute_prototypes(
        embeddings=support_embeddings,
        labels=support_labels,
        num_classes=NUM_CLASSES
    )

    print(
        f"Prototype shape: "
        f"{tuple(prototypes.shape)}"
    )


    # ========================================================
    # 7. Validation
    # ========================================================

    print(
        "\n正在提取 Validation embeddings..."
    )

    val_embeddings, val_labels = extract_embeddings(
        backbone,
        val_loader
    )

    val_acc = evaluate_protonet(
        embeddings=val_embeddings,
        labels=val_labels,
        prototypes=prototypes
    )

    print(
        f"Validation Accuracy: "
        f"{val_acc:.2f}%"
    )


    # ========================================================
    # 8. Test
    # ========================================================

    print(
        "\n正在提取 Test embeddings..."
    )

    test_embeddings, test_labels = extract_embeddings(
        backbone,
        test_loader
    )

    test_acc = evaluate_protonet(
        embeddings=test_embeddings,
        labels=test_labels,
        prototypes=prototypes
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )


    # ========================================================
    # 9. 保存结果
    # ========================================================

    save_results(
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        val_acc=val_acc,
        test_acc=test_acc
    )


    # ========================================================
    # 10. 输出结果
    # ========================================================

    print("\n" + "=" * 70)
    print("Model-level Baseline 实验结束")
    print("=" * 70)

    print(
        "Model: Frozen ImageNet Pretrained ResNet18"
    )

    print(
        "Classifier: ProtoNet"
    )

    print(
        "Embedding: 512-d"
    )

    print(
        "Distance: Squared Euclidean"
    )

    print(
        f"Validation Accuracy: "
        f"{val_acc:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    print("=" * 70)


# ============================================================
# 11. 程序入口
# ============================================================

if __name__ == "__main__":
    main()