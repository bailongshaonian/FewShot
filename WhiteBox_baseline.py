import os
import random
import copy
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100
K_SHOT = 20

BATCH_SIZE = 128
EPOCHS = 100

LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

SEED = 42
NUM_WORKERS = 4

EVAL_FREQ = 5


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

    # 保证实验尽可能可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. 数据增强
# ============================================================

def get_transforms():

    # Data-level baseline：
    # 不使用任何预训练知识，仅通过数据增强扩充有限训练样本
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.2, 1.0)
        ),
        transforms.RandomHorizontalFlip(),

        # 较强的数据增强
        transforms.RandAugment(
            num_ops=2,
            magnitude=9
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集和测试集不进行随机增强
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


# ============================================================
# 4. 构建模型
# ============================================================

def build_model(num_classes):

    # --------------------------------------------------------
    # 随机初始化 ResNet18
    # --------------------------------------------------------
    backbone = resnet18(weights=None)

    feature_dim = backbone.fc.in_features

    # 移除原始 ImageNet 分类头
    backbone.fc = nn.Identity()

    # --------------------------------------------------------
    # Linear Classifier
    #
    # 不使用 Cosine Classifier / Prototype / Metric Learning
    # 避免提前引入 Model-level 结构先验
    # --------------------------------------------------------
    classifier = nn.Linear(
        feature_dim,
        num_classes
    )

    # 组合 backbone + classifier
    model = nn.Sequential(
        backbone,
        classifier
    )

    return model


# ============================================================
# 5. 单轮训练
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    amp_enabled
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------
        with torch.autocast(
            device_type=DEVICE,
            enabled=amp_enabled
        ):

            logits = model(images)

            loss = criterion(
                logits,
                targets
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # ----------------------------------------------------
        # 统计
        # ----------------------------------------------------
        total_loss += loss.item() * targets.size(0)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

    avg_loss = total_loss / total

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# 6. 验证 / 测试
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion=None,
    amp_enabled=False
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE,
            enabled=amp_enabled
        ):

            logits = model(images)

            if criterion is not None:
                loss = criterion(
                    logits,
                    targets
                )

        if criterion is not None:
            total_loss += (
                loss.item() * targets.size(0)
            )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

    accuracy = 100.0 * correct / total

    if criterion is not None:
        avg_loss = total_loss / total
        return avg_loss, accuracy

    return accuracy

# ============================================================
# 结果保存
# ============================================================

def save_results(
    test_acc,
    best_val_acc,
    best_epoch,
    train_size,
    val_size,
    test_size
):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

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
        f.write("White-Box Baseline\n")
        f.write("=" * 70 + "\n")

        # 实验配置
        f.write(f"Seed: {SEED}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Model: Randomly Initialized ResNet18\n")
        f.write(f"Classifier: Linear\n")
        f.write(f"Data Augmentation: RandAugment\n")
        f.write(f"Num Classes: {NUM_CLASSES}\n")
        f.write(f"K-shot: {K_SHOT}\n")

        # 训练配置
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Momentum: {MOMENTUM}\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")

        # 数据规模
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Validation Size: {val_size}\n")
        f.write(f"Test Size: {test_size}\n")

        # 实验结果
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")

        f.write("=" * 70 + "\n\n")

    print(
        f"\n实验结果已保存至: {results_file}"
    )

# ============================================================
# 7. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Data-level Few-shot Baseline")
    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Classes      : {NUM_CLASSES}")
    print(f"K-shot       : {K_SHOT}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Seed         : {SEED}")

    print("\n实验结构：")
    print("Random ResNet18")
    print("      ↓")
    print("Data Augmentation (RandAugment)")
    print("      ↓")
    print("Linear Classifier")
    print("      ↓")
    print("Few-shot Classification")
    print("=" * 70)


    # ========================================================
    # 1. 数据集
    # ========================================================

    train_transform, eval_transform = get_transforms()

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
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )


    # ========================================================
    # 2. 检查数据划分
    # ========================================================

    print("\n数据集信息：")

    print(
        f"Train images : {len(train_dataset)}"
    )

    print(
        f"Val images   : {len(val_dataset)}"
    )

    print(
        f"Test images  : {len(test_dataset)}"
    )

    print(
        f"Train classes: {len(train_dataset.classes)}"
    )

    print(
        f"Val classes  : {len(val_dataset.classes)}"
    )

    print(
        f"Test classes : {len(test_dataset.classes)}"
    )


    # 检查类别名称是否一致
    assert train_dataset.classes == val_dataset.classes, \
        "Train 和 Val 的类别顺序不一致！"

    assert train_dataset.classes == test_dataset.classes, \
        "Train 和 Test 的类别顺序不一致！"

    assert len(train_dataset.classes) == NUM_CLASSES, \
        f"类别数量不是 {NUM_CLASSES}！"


    # ========================================================
    # 3. DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    # 4. 模型
    # ========================================================

    print("\n正在初始化随机权重 ResNet18...")

    model = build_model(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"模型参数量: "
        f"{total_params / 1e6:.2f} M"
    )


    # ========================================================
    # 5. Loss / Optimizer / Scheduler
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )


    # ========================================================
    # 6. AMP
    # ========================================================

    amp_enabled = DEVICE == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )


    # ========================================================
    # 7. 训练
    # ========================================================

    best_val_acc = 0.0
    best_epoch = 0

    # 深拷贝，避免后续训练改变保存的参数
    best_weights = copy.deepcopy(
        model.state_dict()
    )

    print("\n开始训练...\n")

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled
        )

        scheduler.step()


        # ----------------------------------------------------
        # 验证
        # ----------------------------------------------------

        if (
            (epoch + 1) % EVAL_FREQ == 0
            or
            (epoch + 1) == EPOCHS
        ):

            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                amp_enabled=amp_enabled
            )

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                best_epoch = epoch + 1

                best_weights = copy.deepcopy(
                    model.state_dict()
                )

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f} "
                f"| Val Acc: {val_acc:.2f}% "
                f"| LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        else:

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.2f}% "
                f"| LR: {scheduler.get_last_lr()[0]:.6f}"
            )


    # ========================================================
    # 8. 恢复最佳模型
    # ========================================================

    model.load_state_dict(
        best_weights
    )

    print("\n" + "=" * 70)
    print(
        f"最佳验证模型来自 Epoch {best_epoch}"
    )

    print(
        f"Best Val Accuracy: "
        f"{best_val_acc:.2f}%"
    )


    # ========================================================
    # 9. 最终测试
    # ========================================================

    print("\n开始测试集评估...")

    test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=None,
        amp_enabled=amp_enabled
    )

    print("\n" + "=" * 70)
    print("实验结束")
    print("=" * 70)

    print(
        f"[ResNet18 + RandAugment + Linear]"
    )

    print(
        f"{K_SHOT}-shot Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )

    save_results(
        test_acc=test_acc,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset)
    )

    print("=" * 70)


# ============================================================
# 10. 程序入口
# ============================================================

if __name__ == "__main__":
    main()