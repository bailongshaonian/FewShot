import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets

import clip
from tqdm import tqdm


# ============================================================
# 1. 配置参数
# ============================================================

DATA_DIR = "mini-imagenet"

# ImageNet 类别映射
IMAGENET_CLASS_INDEX = os.path.join(
    DATA_DIR,
    "imagenet_class_index.json"
)

# OpenAI CLIP 官方权重
MODEL_PATH = os.path.join(
    "models",
    "vit-b-32.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100
K_SHOT = 20

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 100

LR = 0.001
WEIGHT_DECAY = 1e-4

EVAL_FREQ = 5

# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

NUM_WORKERS = 0

SEED = 42


# ============================================================
# 2. 随机种子
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 3. 加载 CLIP
# ============================================================

def load_clip_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到 CLIP 模型：{MODEL_PATH}\n"
            f"请确认 vit-b-32.pt 位于 models 文件夹中。"
        )

    print("=" * 70)
    print("正在加载 OpenAI CLIP...")
    print("=" * 70)

    print(
        f"Model Path: {MODEL_PATH}"
    )

    print(
        f"Device: {DEVICE}"
    )

    model, preprocess = clip.load(
        MODEL_PATH,
        device=DEVICE,
        jit=False
    )

    model.eval()

    # --------------------------------------------------------
    # 冻结 CLIP
    # --------------------------------------------------------

    for param in model.parameters():
        param.requires_grad = False

    print(
        "CLIP Model: ViT-B/32"
    )

    print(
        "Backbone: Frozen"
    )

    print("=" * 70)

    return model, preprocess


# ============================================================
# 4. 加载数据集
# ============================================================

def load_datasets(preprocess):

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

    # --------------------------------------------------------
    # CLIP 官方 preprocess
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=preprocess
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=preprocess
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=preprocess
    )

    # --------------------------------------------------------
    # 检查类别
    # --------------------------------------------------------

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
    ), (
        f"类别数量不是 {NUM_CLASSES}，"
        f"实际为 {len(train_dataset.classes)}"
    )

    # --------------------------------------------------------
    # 检查 20-shot
    # --------------------------------------------------------

    expected_train_size = (
        NUM_CLASSES * K_SHOT
    )

    if len(train_dataset) != expected_train_size:

        raise ValueError(
            f"训练集大小为 {len(train_dataset)}，"
            f"但 {NUM_CLASSES} 类 × {K_SHOT}-shot "
            f"应该为 {expected_train_size}。"
        )

    print("\n数据集信息：")

    print(
        f"Train images: {len(train_dataset)}"
    )

    print(
        f"Val images  : {len(val_dataset)}"
    )

    print(
        f"Test images : {len(test_dataset)}"
    )

    print(
        f"Classes     : {len(train_dataset.classes)}"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# 5. DataLoader
# ============================================================

def build_dataloader(
    dataset,
    shuffle=False
):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=False
    )


# ============================================================
# 6. 构建 Linear Probe
# ============================================================

def build_classifier(
    feature_dim,
    num_classes
):

    classifier = nn.Linear(
        feature_dim,
        num_classes
    )

    classifier = classifier.to(
        DEVICE
    )

    return classifier


# ============================================================
# 7. 提取 CLIP Image Features
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    loader,
    description
):

    model.eval()

    all_features = []
    all_labels = []

    for images, labels in tqdm(
        loader,
        desc=f"提取 {description} Features"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # CLIP Image Encoder
        # ----------------------------------------------------

        features = model.encode_image(
            images
        )

        # ----------------------------------------------------
        # 转 float32
        # ----------------------------------------------------

        features = features.float()

        all_features.append(
            features.cpu()
        )

        all_labels.append(
            labels
        )

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    return features, labels


# ============================================================
# 8. Linear Classifier Training
# ============================================================

def train_classifier(
    classifier,
    train_features,
    train_labels,
    val_features,
    val_labels
):

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_feature_dataset = torch.utils.data.TensorDataset(
        train_features,
        train_labels
    )

    train_feature_loader = DataLoader(
        train_feature_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    # --------------------------------------------------------
    # Loss / Optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_val_acc = 0.0
    best_epoch = 0

    best_weights = {
        key: value.detach().clone()
        for key, value
        in classifier.state_dict().items()
    }

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\n开始训练 Linear Classifier...\n"
    )

    for epoch in range(EPOCHS):

        classifier.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        for features, labels in train_feature_loader:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = classifier(
                features
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                logits.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_loss = (
            total_loss / total
        )

        train_acc = (
            100.0
            * correct
            / total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if (
            (epoch + 1) % EVAL_FREQ == 0
            or
            (epoch + 1) == EPOCHS
        ):

            classifier.eval()

            with torch.no_grad():

                val_features_device = (
                    val_features.to(DEVICE)
                )

                val_labels_device = (
                    val_labels.to(DEVICE)
                )

                val_logits = classifier(
                    val_features_device
                )

                val_predictions = (
                    val_logits.argmax(dim=1)
                )

                val_acc = (
                    100.0
                    * (
                        val_predictions
                        == val_labels_device
                    ).sum().item()
                    / len(val_labels_device)
                )

            # ------------------------------------------------
            # 保存最佳模型
            # ------------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = val_acc
                best_epoch = epoch + 1

                best_weights = {
                    key: value.detach().clone()
                    for key, value
                    in classifier.state_dict().items()
                }

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.2f}% "
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

    # --------------------------------------------------------
    # 恢复最佳分类器
    # --------------------------------------------------------

    classifier.load_state_dict(
        best_weights
    )

    return (
        classifier,
        best_val_acc,
        best_epoch
    )


# ============================================================
# 9. 最终评估
# ============================================================

@torch.no_grad()
def evaluate_classifier(
    classifier,
    features,
    labels,
    description
):

    classifier.eval()

    features = features.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    logits = classifier(
        features
    )

    predictions = logits.argmax(
        dim=1
    )

    accuracy = (
        100.0
        * (
            predictions == labels
        ).sum().item()
        / len(labels)
    )

    # --------------------------------------------------------
    # Mean Class Accuracy
    # --------------------------------------------------------

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
        device=DEVICE
    )

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
        device=DEVICE
    )

    for class_id in range(
        NUM_CLASSES
    ):

        mask = (
            labels == class_id
        )

        if mask.any():

            class_total[class_id] += (
                mask.sum()
            )

            class_correct[class_id] += (
                (
                    predictions[mask]
                    == labels[mask]
                )
                .sum()
            )

    valid_classes = (
        class_total > 0
    )

    class_accuracy = (
        class_correct[valid_classes].float()
        /
        class_total[valid_classes].float()
    )

    mean_class_accuracy = (
        class_accuracy.mean().item()
        * 100.0
    )

    print(
        f"{description} Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"{description} Mean Class Accuracy: "
        f"{mean_class_accuracy:.2f}%"
    )

    return (
        accuracy,
        mean_class_accuracy
    )


# ============================================================
# 10. 保存实验结果
# ============================================================

def save_results(
    best_val_acc,
    best_epoch,
    test_acc,
    val_mean_class_acc,
    test_mean_class_acc,
    train_size,
    val_size,
    test_size,
    feature_dim
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
            "CLIP Linear Probe Baseline\n"
        )
        f.write("=" * 70 + "\n")

        # ----------------------------------------------------
        # 实验信息
        # ----------------------------------------------------

        f.write(
            "Experiment: clip_linear_probe\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        # ----------------------------------------------------
        # CLIP
        # ----------------------------------------------------

        f.write(
            "Backbone: OpenAI CLIP ViT-B/32\n"
        )

        f.write(
            f"Model Path: {MODEL_PATH}\n"
        )

        f.write(
            "Backbone Frozen: Yes\n"
        )

        f.write(
            "Fine-tuning CLIP Backbone: No\n"
        )

        f.write(
            f"Image Feature Dimension: "
            f"{feature_dim}\n"
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        f.write(
            "Classifier: Linear\n"
        )

        f.write(
            "Classifier Training: Yes\n"
        )

        # ----------------------------------------------------
        # Few-shot
        # ----------------------------------------------------

        f.write(
            f"Num Classes: {NUM_CLASSES}\n"
        )

        f.write(
            f"K-shot: {K_SHOT}\n"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        f.write(
            f"Epochs: {EPOCHS}\n"
        )

        f.write(
            f"Batch Size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Learning Rate: {LR}\n"
        )

        f.write(
            f"Weight Decay: {WEIGHT_DECAY}\n"
        )

        f.write(
            f"Evaluation Frequency: "
            f"{EVAL_FREQ}\n"
        )

        # ----------------------------------------------------
        # 数据
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
        # Results
        # ----------------------------------------------------

        f.write(
            f"Best Validation Accuracy: "
            f"{best_val_acc:.2f}%\n"
        )

        f.write(
            f"Best Epoch: {best_epoch}\n"
        )

        f.write(
            f"Validation Mean Class Accuracy: "
            f"{val_mean_class_acc:.2f}%\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_acc:.2f}%\n"
        )

        f.write(
            f"Test Mean Class Accuracy: "
            f"{test_mean_class_acc:.2f}%\n"
        )

        f.write("\n")

    print(
        f"\n实验结果已追加保存至："
        f"{results_file}"
    )


# ============================================================
# 11. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("CLIP Linear Probe Baseline")
    print("=" * 70)

    print(
        "CLIP ViT-B/32"
    )

    print(
        "      ↓"
    )

    print(
        "Frozen Image Encoder"
    )

    print(
        "      ↓"
    )

    print(
        "Image Feature"
    )

    print(
        "      ↓"
    )

    print(
        "Train Linear Classifier"
    )

    print(
        "      ↓"
    )

    print(
        "100-class Prediction"
    )

    print("=" * 70)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"K-shot       : {K_SHOT}"
    )

    print(
        f"Batch Size   : {BATCH_SIZE}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Learning Rate: {LR}"
    )


    # ========================================================
    # 1. 加载 CLIP
    # ========================================================

    model, preprocess = (
        load_clip_model()
    )


    # ========================================================
    # 2. 数据
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets(
        preprocess
    )


    # ========================================================
    # 3. DataLoader
    # ========================================================

    train_loader = build_dataloader(
        train_dataset,
        shuffle=False
    )

    val_loader = build_dataloader(
        val_dataset,
        shuffle=False
    )

    test_loader = build_dataloader(
        test_dataset,
        shuffle=False
    )


    # ========================================================
    # 4. 提取 CLIP Features
    #
    # 因为 CLIP 冻结，所以只需要提取一次。
    # ========================================================

    print(
        "\n开始提取 CLIP Image Features..."
    )

    train_features, train_labels = (
        extract_features(
            model,
            train_loader,
            "Train"
        )
    )

    val_features, val_labels = (
        extract_features(
            model,
            val_loader,
            "Validation"
        )
    )

    test_features, test_labels = (
        extract_features(
            model,
            test_loader,
            "Test"
        )
    )

    feature_dim = train_features.shape[1]

    print(
        "\nFeature Shapes:"
    )

    print(
        f"Train: {tuple(train_features.shape)}"
    )

    print(
        f"Val  : {tuple(val_features.shape)}"
    )

    print(
        f"Test : {tuple(test_features.shape)}"
    )


    # ========================================================
    # 5. 构建 Linear Classifier
    # ========================================================

    classifier = build_classifier(
        feature_dim=feature_dim,
        num_classes=NUM_CLASSES
    )


    # ========================================================
    # 6. 训练 Linear Classifier
    # ========================================================

    (
        classifier,
        best_val_acc,
        best_epoch
    ) = train_classifier(
        classifier=classifier,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels
    )


    # ========================================================
    # 7. Validation
    # ========================================================

    (
        final_val_acc,
        val_mean_class_acc
    ) = evaluate_classifier(
        classifier,
        val_features,
        val_labels,
        "Validation"
    )


    # ========================================================
    # 8. Test
    # ========================================================

    (
        test_acc,
        test_mean_class_acc
    ) = evaluate_classifier(
        classifier,
        test_features,
        test_labels,
        "Test"
    )


    # ========================================================
    # 9. 保存结果
    # ========================================================

    save_results(
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        test_acc=test_acc,
        val_mean_class_acc=val_mean_class_acc,
        test_mean_class_acc=test_mean_class_acc,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        feature_dim=feature_dim
    )


    # ========================================================
    # 10. 最终输出
    # ========================================================

    print("\n" + "=" * 70)
    print("CLIP Linear Probe 实验结束")
    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Final Validation Accuracy: "
        f"{final_val_acc:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    print("=" * 70)


# ============================================================
# 12. 程序入口
# ============================================================

if __name__ == "__main__":
    main()