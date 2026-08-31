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

    # 为了尽可能保证实验复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. 数据增强
# ============================================================

def get_transforms():

    # 与 White-box baseline 保持一致
    # 唯一核心变化：Backbone 改为 ImageNet 预训练 ResNet18
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.2, 1.0)
        ),
        transforms.RandomHorizontalFlip(),
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
# 4. 构建预训练 ResNet18 + Linear
# ============================================================

def build_model(num_classes):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到预训练模型：{MODEL_PATH}\n"
            f"请先运行 download_pretrained.py"
        )

    print("\n正在加载 ImageNet 预训练 ResNet18...")

    # 先建立与保存权重一致的 ResNet18
    backbone = resnet18(weights=None)

    # 加载本地预训练权重
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    backbone.load_state_dict(state_dict)

    # 保存原始 feature dimension
    feature_dim = backbone.fc.in_features

    # 去掉 ImageNet 原始分类头
    backbone.fc = nn.Identity()

    # 当前 Few-shot 任务自己的分类头
    classifier = nn.Linear(
        feature_dim,
        num_classes
    )

    # 组合成完整模型
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

        total_loss += (
            loss.item() * targets.size(0)
        )

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
# 7. 保存实验结果
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
            "Data-level Prior Baseline\n"
        )
        f.write("=" * 70 + "\n")

        # ----------------------------------------------------
        # 实验身份
        # ----------------------------------------------------

        f.write(
            "Experiment: data_baseline_pretrained_resnet18\n"
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
            "Classifier: Linear\n"
        )

        f.write(
            "Pretrained Weights: "
            f"{MODEL_PATH}\n"
        )

        # ----------------------------------------------------
        # Data-level
        # ----------------------------------------------------

        f.write(
            "Data Augmentation: "
            "RandomResizedCrop + "
            "RandomHorizontalFlip + "
            "RandAugment\n"
        )

        f.write(
            f"Num Classes: {NUM_CLASSES}\n"
        )

        f.write(
            f"K-shot: {K_SHOT}\n"
        )

        # ----------------------------------------------------
        # 训练配置
        # ----------------------------------------------------

        f.write(
            f"Batch Size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Epochs: {EPOCHS}\n"
        )

        f.write(
            f"Learning Rate: {LR}\n"
        )

        f.write(
            f"Momentum: {MOMENTUM}\n"
        )

        f.write(
            f"Weight Decay: {WEIGHT_DECAY}\n"
        )

        # ----------------------------------------------------
        # 数据规模
        # ----------------------------------------------------

        f.write(
            f"Train Size: {train_size}\n"
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
            f"Best Validation Accuracy: "
            f"{best_val_acc:.2f}%\n"
        )

        f.write(
            f"Best Epoch: {best_epoch}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_acc:.2f}%\n"
        )

        f.write("\n")


    print(
        f"\n实验结果已追加保存至："
        f"{results_file}"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Data-level Prior Baseline")
    print("=" * 70)

    print(
        "ImageNet Pretrained ResNet18"
    )

    print(
        "        ↓"
    )

    print(
        "Data Augmentation"
    )

    print(
        "        ↓"
    )

    print(
        "Linear Classifier"
    )

    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Classes      : {NUM_CLASSES}")
    print(f"K-shot       : {K_SHOT}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Seed         : {SEED}")


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
    # 2. 检查数据
    # ========================================================

    print("\n数据集信息：")

    print(
        f"Train images : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val images   : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test images  : "
        f"{len(test_dataset)}"
    )

    print(
        f"Train classes: "
        f"{len(train_dataset.classes)}"
    )

    print(
        f"Val classes  : "
        f"{len(val_dataset.classes)}"
    )

    print(
        f"Test classes : "
        f"{len(test_dataset.classes)}"
    )


    # 类别映射必须一致
    assert (
        train_dataset.classes
        == val_dataset.classes
    ), "Train 与 Val 的类别顺序不一致！"

    assert (
        train_dataset.classes
        == test_dataset.classes
    ), "Train 与 Test 的类别顺序不一致！"

    assert (
        len(train_dataset.classes)
        == NUM_CLASSES
    ), f"类别数量不是 {NUM_CLASSES}！"


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

    model = build_model(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n模型参数量: "
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
    # 7. 训练循环
    # ========================================================

    best_val_acc = 0.0
    best_epoch = 0

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    print("\n开始 Data-level Baseline 训练...\n")

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
        f"最佳模型 Epoch: {best_epoch}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )


    # ========================================================
    # 9. 最终 Test
    # ========================================================

    print("\n开始测试集评估...")

    test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=None,
        amp_enabled=amp_enabled
    )

    print("\n" + "=" * 70)
    print("Data-level Baseline 实验结束")
    print("=" * 70)

    print(
        "Model: ImageNet Pretrained ResNet18"
    )

    print(
        f"{K_SHOT}-shot Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )

    # ========================================================
    # 10. 保存结果
    # ========================================================

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
# 11. 程序入口
# ============================================================

if __name__ == "__main__":
    main()