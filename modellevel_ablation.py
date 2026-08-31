import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms
from torchvision.models import resnet18

from tqdm import tqdm


# ============================================================
# 1. 配置
# ============================================================

DATA_DIR = "mini-imagenet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100
K_SHOT = 20

# Episodic Training
N_WAY = 5
N_SHOT = 5
N_QUERY = 10

EPISODES_PER_EPOCH = 50
EPOCHS = 100

# 验证频率
EVAL_FREQ = 5

# Embedding 提取
BATCH_SIZE = 128
NUM_WORKERS = 4

# Optimizer
LR = 0.001
WEIGHT_DECAY = 5e-4

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. 数据增强
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.5, 1.0)
        ),

        transforms.RandomHorizontalFlip(),

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
# 4. 构建随机 ResNet18
# ============================================================

def build_model():

    print("\n正在初始化随机权重 ResNet18...")

    model = resnet18(weights=None)

    feature_dim = model.fc.in_features

    model.fc = nn.Identity()

    model = model.to(DEVICE)

    print(
        f"Embedding Dimension: {feature_dim}"
    )

    return model


# ============================================================
# 5. 构建类别索引
# ============================================================

def build_class_indices(dataset):

    class_indices = {
        class_id: []
        for class_id in range(NUM_CLASSES)
    }

    for index, (_, label) in enumerate(
        dataset.samples
    ):
        class_indices[label].append(index)

    for class_id in range(NUM_CLASSES):

        num_samples = len(
            class_indices[class_id]
        )

        if num_samples < K_SHOT:

            raise ValueError(
                f"类别 {class_id} "
                f"只有 {num_samples} 张图像，"
                f"不足 {K_SHOT}-shot。"
            )

    return class_indices


# ============================================================
# 6. Episode Dataset
# ============================================================

class EpisodeDataset(Dataset):

    """
    专门用于生成 episodic task。

    每个 item 返回：
        selected_classes
        support_indices
        query_indices
    """

    def __init__(
        self,
        class_indices,
        num_classes,
        n_way,
        n_shot,
        n_query,
        episodes
    ):

        self.class_indices = class_indices
        self.num_classes = num_classes

        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        self.episodes = episodes

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):

        selected_classes = random.sample(
            range(self.num_classes),
            self.n_way
        )

        support_indices = []
        query_indices = []

        for class_id in selected_classes:

            available = self.class_indices[
                class_id
            ]

            selected = random.sample(
                available,
                self.n_shot + self.n_query
            )

            support_indices.extend(
                selected[:self.n_shot]
            )

            query_indices.extend(
                selected[self.n_shot:]
            )

        return (
            selected_classes,
            support_indices,
            query_indices
        )


# ============================================================
# 7. Episode Collate
# ============================================================

def episode_collate_fn(batch):

    # batch_size 固定为 1
    return batch[0]


# ============================================================
# 8. 获取 Episode 图像
# ============================================================

def load_episode_images(
    dataset,
    support_indices,
    query_indices
):

    support_images = []
    support_labels = []

    query_images = []
    query_labels = []

    for index in support_indices:

        image, label = dataset[index]

        support_images.append(image)
        support_labels.append(label)

    for index in query_indices:

        image, label = dataset[index]

        query_images.append(image)
        query_labels.append(label)

    support_images = torch.stack(
        support_images
    )

    query_images = torch.stack(
        query_images
    )

    support_labels = torch.tensor(
        support_labels,
        dtype=torch.long
    )

    query_labels = torch.tensor(
        query_labels,
        dtype=torch.long
    )

    return (
        support_images,
        support_labels,
        query_images,
        query_labels
    )


# ============================================================
# 9. Episode Prototype
# ============================================================

def compute_episode_prototypes(
    support_embeddings,
    support_labels,
    selected_classes
):

    prototypes = []

    for class_id in selected_classes:

        mask = (
            support_labels == class_id
        )

        class_embeddings = (
            support_embeddings[mask]
        )

        prototype = (
            class_embeddings.mean(dim=0)
        )

        prototypes.append(
            prototype
        )

    return torch.stack(prototypes)


# ============================================================
# 10. ProtoNet logits
# ============================================================

def compute_logits(
    query_embeddings,
    prototypes
):

    distances = torch.cdist(
        query_embeddings,
        prototypes,
        p=2
    ).pow(2)

    return -distances


# ============================================================
# 11. Episode label mapping
# ============================================================

def map_labels(
    labels,
    selected_classes
):

    mapping = {
        class_id: episode_id
        for episode_id, class_id
        in enumerate(selected_classes)
    }

    return torch.tensor(
        [
            mapping[int(label)]
            for label in labels
        ],
        dtype=torch.long,
        device=DEVICE
    )


# ============================================================
# 12. 单个 episode
# ============================================================

def run_episode(
    model,
    dataset,
    episode,
    optimizer,
    criterion,
    scaler,
    amp_enabled
):

    selected_classes, support_indices, query_indices = episode

    (
        support_images,
        support_labels,
        query_images,
        query_labels
    ) = load_episode_images(
        dataset,
        support_indices,
        query_indices
    )

    support_images = support_images.to(
        DEVICE,
        non_blocking=True
    )

    query_images = query_images.to(
        DEVICE,
        non_blocking=True
    )

    support_labels = support_labels.to(
        DEVICE
    )

    query_labels = query_labels.to(
        DEVICE
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    with torch.autocast(
        device_type=DEVICE,
        enabled=amp_enabled
    ):

        support_embeddings = model(
            support_images
        )

        query_embeddings = model(
            query_images
        )

        prototypes = compute_episode_prototypes(
            support_embeddings,
            support_labels,
            selected_classes
        )

        logits = compute_logits(
            query_embeddings,
            prototypes
        )

        episode_labels = map_labels(
            query_labels,
            selected_classes
        )

        loss = criterion(
            logits,
            episode_labels
        )

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    predictions = logits.argmax(
        dim=1
    )

    accuracy = (
        predictions == episode_labels
    ).float().mean().item()

    return (
        loss.item(),
        accuracy
    )


# ============================================================
# 13. Episode Training
# ============================================================

def train_epoch(
    model,
    dataset,
    episode_loader,
    optimizer,
    criterion,
    scaler,
    amp_enabled
):

    model.train()

    total_loss = 0.0
    total_acc = 0.0

    for episode in episode_loader:

        loss, acc = run_episode(
            model=model,
            dataset=dataset,
            episode=episode,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            amp_enabled=amp_enabled
        )

        total_loss += loss
        total_acc += acc

    avg_loss = (
        total_loss
        / len(episode_loader)
    )

    avg_acc = (
        total_acc
        / len(episode_loader)
    ) * 100.0

    return avg_loss, avg_acc


# ============================================================
# 14. Batch Embedding 提取
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    dataset
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    model.eval()

    embeddings_list = []
    labels_list = []

    amp_enabled = DEVICE == "cuda"

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE,
            enabled=amp_enabled
        ):

            embeddings = model(
                images
            )

        embeddings_list.append(
            embeddings.float().cpu()
        )

        labels_list.append(
            labels
        )

    embeddings = torch.cat(
        embeddings_list,
        dim=0
    )

    labels = torch.cat(
        labels_list,
        dim=0
    )

    return embeddings, labels


# ============================================================
# 15. 构建完整 Prototype
# ============================================================

def compute_full_prototypes(
    embeddings,
    labels
):

    feature_dim = embeddings.shape[1]

    prototypes = torch.zeros(
        NUM_CLASSES,
        feature_dim,
        dtype=embeddings.dtype
    )

    for class_id in range(
        NUM_CLASSES
    ):

        mask = (
            labels == class_id
        )

        if not mask.any():

            raise ValueError(
                f"类别 {class_id} "
                f"没有样本。"
            )

        prototypes[class_id] = (
            embeddings[mask].mean(
                dim=0
            )
        )

    return prototypes


# ============================================================
# 16. Prototype Evaluation
# ============================================================

@torch.no_grad()
def evaluate_with_prototypes(
    embeddings,
    labels,
    prototypes
):

    distances = torch.cdist(
        embeddings,
        prototypes,
        p=2
    ).pow(2)

    predictions = distances.argmin(
        dim=1
    )

    accuracy = (
        predictions == labels
    ).float().mean().item()

    return accuracy * 100.0


# ============================================================
# 17. 保存结果
# ============================================================

def save_results(
    best_val_acc,
    best_epoch,
    test_acc,
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
            "From-scratch ProtoNet Baseline\n"
        )
        f.write("=" * 70 + "\n")

        f.write(
            "Experiment: from_scratch_protonet\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            "Model: Randomly Initialized ResNet18\n"
        )

        f.write(
            "Backbone: Trainable\n"
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
            "Pretrained Weights: None\n"
        )

        f.write(
            f"N-way: {N_WAY}\n"
        )

        f.write(
            f"N-shot: {N_SHOT}\n"
        )

        f.write(
            f"N-query: {N_QUERY}\n"
        )

        f.write(
            f"Episodes per Epoch: "
            f"{EPISODES_PER_EPOCH}\n"
        )

        f.write(
            f"Epochs: {EPOCHS}\n"
        )

        f.write(
            f"Evaluation Frequency: "
            f"{EVAL_FREQ}\n"
        )

        f.write(
            f"Learning Rate: {LR}\n"
        )

        f.write(
            f"Weight Decay: {WEIGHT_DECAY}\n"
        )

        f.write(
            f"Train Size: {train_size}\n"
        )

        f.write(
            f"Validation Size: {val_size}\n"
        )

        f.write(
            f"Test Size: {test_size}\n"
        )

        f.write(
            "Episodic Meta-training: Yes\n"
        )

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
# 18. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Optimized From-scratch ProtoNet Baseline")
    print("=" * 70)

    print(
        "Random ResNet18"
    )

    print("        ↓")

    print(
        "Episodic ProtoNet Training"
    )

    print("        ↓")

    print(
        "Learned Embedding"
    )

    print("        ↓")

    print(
        "Prototype Classification"
    )

    print("=" * 70)

    print(f"Device      : {DEVICE}")
    print(f"Classes     : {NUM_CLASSES}")
    print(f"K-shot      : {K_SHOT}")
    print(f"N-way       : {N_WAY}")
    print(f"N-shot      : {N_SHOT}")
    print(f"N-query     : {N_QUERY}")
    print(
        f"Episodes/Epoch: "
        f"{EPISODES_PER_EPOCH}"
    )
    print(f"Epochs      : {EPOCHS}")
    print(f"Eval Freq   : {EVAL_FREQ}")
    print(f"LR          : {LR}")
    print(f"Seed        : {SEED}")


    # ========================================================
    # 1. 数据
    # ========================================================

    train_transform, eval_transform = (
        get_transforms()
    )

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
    # 2. 数据检查
    # ========================================================

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

    assert (
        train_dataset.classes
        == val_dataset.classes
    )

    assert (
        train_dataset.classes
        == test_dataset.classes
    )

    assert (
        len(train_dataset.classes)
        == NUM_CLASSES
    )


    # ========================================================
    # 3. 类别索引
    # ========================================================

    class_indices = build_class_indices(
        train_dataset
    )


    # ========================================================
    # 4. Episode Loader
    # ========================================================

    episode_dataset = EpisodeDataset(
        class_indices=class_indices,
        num_classes=NUM_CLASSES,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        episodes=EPISODES_PER_EPOCH
    )

    episode_loader = DataLoader(
        episode_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=episode_collate_fn,
        persistent_workers=(NUM_WORKERS > 0)
    )


    # ========================================================
    # 5. 模型
    # ========================================================

    model = build_model()


    # ========================================================
    # 6. Loss / Optimizer
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # 7. AMP
    # ========================================================

    amp_enabled = DEVICE == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )


    # ========================================================
    # 8. 最佳模型
    # ========================================================

    best_val_acc = 0.0
    best_epoch = 0

    best_weights = copy.deepcopy(
        model.state_dict()
    )


    # ========================================================
    # 9. Training
    # ========================================================

    print(
        "\n开始 Episodic ProtoNet Training...\n"
    )

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_epoch(
            model=model,
            dataset=train_dataset,
            episode_loader=episode_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            amp_enabled=amp_enabled
        )


        # ----------------------------------------------------
        # 只在指定 epoch 做验证
        # ----------------------------------------------------

        if (
            (epoch + 1) % EVAL_FREQ == 0
            or
            (epoch + 1) == EPOCHS
        ):

            # 使用 evaluation transform
            eval_train_dataset = datasets.ImageFolder(
                train_dir,
                transform=eval_transform
            )

            train_embeddings, train_labels = (
                extract_embeddings(
                    model,
                    eval_train_dataset
                )
            )

            prototypes = (
                compute_full_prototypes(
                    train_embeddings,
                    train_labels
                )
            )

            val_embeddings, val_labels = (
                extract_embeddings(
                    model,
                    val_dataset
                )
            )

            val_acc = evaluate_with_prototypes(
                val_embeddings,
                val_labels,
                prototypes
            )

            if val_acc > best_val_acc:

                best_val_acc = val_acc
                best_epoch = epoch + 1

                best_weights = copy.deepcopy(
                    model.state_dict()
                )

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Episode Loss: {train_loss:.4f} "
                f"| Episode Acc: {train_acc:.2f}% "
                f"| Val Acc: {val_acc:.2f}%"
            )

        else:

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Episode Loss: {train_loss:.4f} "
                f"| Episode Acc: {train_acc:.2f}%"
            )


    # ========================================================
    # 10. 恢复最佳模型
    # ========================================================

    model.load_state_dict(
        best_weights
    )

    print("\n" + "=" * 70)

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )


    # ========================================================
    # 11. 最终 Prototype
    # ========================================================

    print(
        "\n重新计算最终 100 类 Prototype..."
    )

    eval_train_dataset = datasets.ImageFolder(
        train_dir,
        transform=eval_transform
    )

    train_embeddings, train_labels = (
        extract_embeddings(
            model,
            eval_train_dataset
        )
    )

    prototypes = compute_full_prototypes(
        train_embeddings,
        train_labels
    )


    # ========================================================
    # 12. Test
    # ========================================================

    print(
        "\n开始测试集评估..."
    )

    test_embeddings, test_labels = (
        extract_embeddings(
            model,
            test_dataset
        )
    )

    test_acc = evaluate_with_prototypes(
        test_embeddings,
        test_labels,
        prototypes
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )


    # ========================================================
    # 13. 保存
    # ========================================================

    save_results(
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        test_acc=test_acc,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset)
    )


    # ========================================================
    # 14. 最终输出
    # ========================================================

    print("\n" + "=" * 70)
    print("From-scratch ProtoNet 实验结束")
    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    print("=" * 70)


# ============================================================
# 19. 程序入口
# ============================================================

if __name__ == "__main__":
    main()