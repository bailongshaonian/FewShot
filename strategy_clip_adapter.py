import os
import json
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

import clip
from tqdm import tqdm


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = "mini-imagenet"

MODEL_PATH = os.path.join(
    "models",
    "vit-b-32.pt"
)

CLASS_INDEX_PATH = os.path.join(
    DATA_DIR,
    "imagenet_class_index.json"
)

RESULT_DIR = "results"
RESULT_FILE = os.path.join(
    RESULT_DIR,
    "results_data.txt"
)

CHECKPOINT_PATH = os.path.join(
    "models",
    "clip_adapter_best.pth"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 42


# ============================================================
# Dataset
# ============================================================

NUM_CLASSES = 100
K_SHOT = 20

BATCH_SIZE = 64
NUM_WORKERS = 0


# ============================================================
# Training
# ============================================================

EPOCHS = 50

LR = 1e-3

WEIGHT_DECAY = 1e-4

EVAL_FREQ = 5


# ============================================================
# CLIP-Adapter
# ============================================================

# 官方实现中 reduction=4
ADAPTER_REDUCTION = 4

# ViT-B/32 image feature:
# 512 -> 128 -> 512
#
# 如果模型返回不同维度，代码会自动按照
# feature_dim // ADAPTER_REDUCTION 计算。

RESIDUAL_RATIO = 0.2


# ============================================================
# Prompt
# ============================================================

PROMPT_TEMPLATE = "a photo of a {}"


# ============================================================
# 2. Random Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 为了实验稳定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Load CLIP
# ============================================================

def load_clip_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"找不到 CLIP 模型：{MODEL_PATH}"
        )

    print("=" * 70)
    print("Loading OpenAI CLIP")
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

    # --------------------------------------------------------
    # 官方 CLIP-Adapter 实现会显式转 FP32
    # --------------------------------------------------------

    model = model.float()

    model.eval()

    # --------------------------------------------------------
    # 冻结 CLIP
    # --------------------------------------------------------

    for param in model.parameters():

        param.requires_grad = False

    print(
        "Model: OpenAI CLIP ViT-B/32"
    )

    print(
        f"Model dtype: {model.dtype}"
    )

    print(
        "CLIP Backbone: Frozen"
    )

    print("=" * 70)

    return model, preprocess


# ============================================================
# 4. ImageNet class mapping
# ============================================================

def load_class_names():

    if not os.path.exists(
        CLASS_INDEX_PATH
    ):

        raise FileNotFoundError(
            f"找不到：{CLASS_INDEX_PATH}"
        )

    with open(
        CLASS_INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_index = json.load(f)

    class_names = []

    for i in range(NUM_CLASSES):

        key = str(i)

        if key not in class_index:

            raise KeyError(
                f"ImageNet class index 中不存在 {key}"
            )

        wnid, name = class_index[key]

        name = (
            name
            .replace("_", " ")
            .replace("-", " ")
        )

        class_names.append(name)

    return class_names


# ============================================================
# 5. Dataset
# ============================================================

def load_datasets(
    preprocess
):

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
    # 类别检查
    # --------------------------------------------------------

    if train_dataset.classes != val_dataset.classes:

        raise ValueError(
            "Train 与 Val 的类别顺序不一致！"
        )

    if train_dataset.classes != test_dataset.classes:

        raise ValueError(
            "Train 与 Test 的类别顺序不一致！"
        )

    if len(train_dataset.classes) != NUM_CLASSES:

        raise ValueError(
            f"类别数量为 {len(train_dataset.classes)}，"
            f"而预期为 {NUM_CLASSES}"
        )

    # --------------------------------------------------------
    # 当前实验设定：
    # Train = 100 classes × 20 images
    # --------------------------------------------------------

    expected_train_size = (
        NUM_CLASSES * K_SHOT
    )

    if len(train_dataset) != expected_train_size:

        raise ValueError(
            f"训练集数量为 {len(train_dataset)}，"
            f"而当前 K-shot 设置要求 "
            f"{expected_train_size}"
        )

    print(
        "\nDataset Information:"
    )

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Val  : {len(val_dataset)}"
    )

    print(
        f"Test : {len(test_dataset)}"
    )

    print(
        f"Classes: {len(train_dataset.classes)}"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# 6. DataLoader
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
# 7. Extract Frozen CLIP Image Features
# ============================================================

@torch.no_grad()
def extract_clip_image_features(
    model,
    dataset,
    description
):

    loader = build_dataloader(
        dataset,
        shuffle=False
    )

    all_features = []

    all_labels = []

    print(
        f"\nExtracting {description} "
        f"CLIP Image Features..."
    )

    for images, labels in tqdm(
        loader,
        desc=f"{description} Features"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        features = model.encode_image(
            images
        )

        # 官方 Adapter 使用 FP32
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

    print(
        f"{description} feature shape: "
        f"{tuple(features.shape)}"
    )

    return (
        features,
        labels
    )


# ============================================================
# 8. Build Text Features
# ============================================================

@torch.no_grad()
def build_text_features(
    model,
    class_names
):

    prompts = [
        PROMPT_TEMPLATE.format(
            class_name
        )
        for class_name in class_names
    ]

    print(
        "\nPrompt examples:"
    )

    for prompt in prompts[:10]:

        print(
            f"  {prompt}"
        )

    tokens = clip.tokenize(
        prompts
    ).to(DEVICE)

    text_features = model.encode_text(
        tokens
    )

    text_features = (
        text_features.float()
    )

    # CLIP cosine similarity
    text_features = F.normalize(
        text_features,
        dim=-1
    )

    print(
        f"Text feature shape: "
        f"{tuple(text_features.shape)}"
    )

    return text_features


# ============================================================
# 9. CLIP Adapter
# ============================================================

class CLIPAdapter(nn.Module):

    """
    CLIP-Adapter

    官方实现的核心结构：

        c_in
          ↓
      c_in / reduction
          ↓
         ReLU
          ↓
         c_in
          ↓
         ReLU

    然后：

        output =
            ratio * adapter(x)
            +
            (1-ratio) * x

    这里：

        reduction = 4
        ratio = 0.2

    所以 ViT-B/32 的典型情况：

        512
         ↓
        128
         ↓
        512
    """

    def __init__(
        self,
        feature_dim,
        reduction=4,
        ratio=0.2
    ):

        super().__init__()

        if feature_dim <= 0:

            raise ValueError(
                "feature_dim 必须大于 0"
            )

        bottleneck_dim = (
            feature_dim // reduction
        )

        if bottleneck_dim <= 0:

            raise ValueError(
                "bottleneck dimension <= 0"
            )

        self.feature_dim = feature_dim

        self.bottleneck_dim = (
            bottleneck_dim
        )

        self.reduction = reduction

        self.ratio = ratio

        # ----------------------------------------------------
        # 官方 CLIP-Adapter 结构
        # ----------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                feature_dim,
                bottleneck_dim,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Linear(
                bottleneck_dim,
                feature_dim,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x
    ):

        adapted_features = self.fc(
            x
        )

        # ----------------------------------------------------
        # Residual Feature Blending
        #
        # 重要：
        #
        # 不是：
        #
        # x + ratio * adapter(x)
        #
        # 而是：
        #
        # ratio * adapter(x)
        # +
        # (1-ratio) * x
        # ----------------------------------------------------

        output = (
            self.ratio
            * adapted_features
            +
            (1.0 - self.ratio)
            * x
        )

        return output


# ============================================================
# 10. CLIP Adapter Model
# ============================================================

class CLIPAdapterModel(nn.Module):

    """
    这里只对 Image Feature 做 Adapter。

    CLIP Image Encoder:
        Frozen

    CLIP Text Encoder:
        Frozen

    Adapter:
        Trainable
    """

    def __init__(
        self,
        clip_model,
        feature_dim,
        reduction=4,
        ratio=0.2
    ):

        super().__init__()

        self.clip_model = (
            clip_model
        )

        self.adapter = (
            CLIPAdapter(
                feature_dim=feature_dim,
                reduction=reduction,
                ratio=ratio
            )
        )

        # ----------------------------------------------------
        # 再次确保 CLIP 冻结
        # ----------------------------------------------------

        for param in (
            self.clip_model.parameters()
        ):

            param.requires_grad = False

    def forward(
        self,
        image_features
    ):

        # ----------------------------------------------------
        # Adapter
        #
        # image_features 已经来自
        # Frozen CLIP Image Encoder
        # ----------------------------------------------------

        adapted_features = (
            self.adapter(
                image_features
            )
        )

        # ----------------------------------------------------
        # Normalize AFTER residual blending
        # ----------------------------------------------------

        adapted_features = F.normalize(
            adapted_features,
            dim=-1
        )

        return adapted_features


# ============================================================
# 11. Training Dataset
# ============================================================

def build_feature_loader(
    features,
    labels,
    shuffle=True
):

    dataset = TensorDataset(
        features,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    return loader


# ============================================================
# 12. Evaluate Adapter
# ============================================================

@torch.no_grad()
def evaluate_adapter(
    adapter_model,
    image_features,
    labels,
    text_features,
    logit_scale
):

    adapter_model.eval()

    image_features = image_features.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Adapter
    # --------------------------------------------------------

    adapted_features = (
        adapter_model(
            image_features
        )
    )

    # --------------------------------------------------------
    # CLIP similarity
    # --------------------------------------------------------

    logits = (
        logit_scale
        * adapted_features
        @ text_features.T
    )

    predictions = (
        logits.argmax(
            dim=1
        )
    )

    # --------------------------------------------------------
    # Overall Accuracy
    # --------------------------------------------------------

    accuracy = (
        100.0
        * (
            predictions == labels
        ).sum().item()
        /
        len(labels)
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
        class_correct[
            valid_classes
        ].float()
        /
        class_total[
            valid_classes
        ].float()
    )

    mean_class_accuracy = (
        class_accuracy.mean().item()
        * 100.0
    )

    return (
        accuracy,
        mean_class_accuracy
    )


# ============================================================
# 13. Training
# ============================================================

def train_adapter(
    adapter_model,
    train_features,
    train_labels,
    val_features,
    val_labels,
    text_features,
    logit_scale
):

    train_loader = build_feature_loader(
        train_features,
        train_labels,
        shuffle=True
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # 只有 Adapter 参数
    # --------------------------------------------------------

    trainable_parameters = [
        param
        for param in adapter_model.parameters()
        if param.requires_grad
    ]

    if len(trainable_parameters) == 0:

        raise RuntimeError(
            "没有找到可训练的 Adapter 参数！"
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    best_val_acc = -1.0

    best_epoch = 0

    best_adapter_state = (
        copy.deepcopy(
            adapter_model.adapter.state_dict()
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Starting CLIP-Adapter Training"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        adapter_model.train()

        total_loss = 0.0

        total_correct = 0

        total_samples = 0

        for features, labels in train_loader:

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

            # ------------------------------------------------
            # Adapter
            # ------------------------------------------------

            adapted_features = (
                adapter_model(
                    features
                )
            )

            # ------------------------------------------------
            # Classification logits
            # ------------------------------------------------

            logits = (
                logit_scale
                * adapted_features
                @ text_features.T
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            batch_size = (
                labels.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_correct += (
                (
                    logits.argmax(dim=1)
                    == labels
                )
                .sum()
                .item()
            )

            total_samples += (
                batch_size
            )

        scheduler.step()

        train_loss = (
            total_loss
            / total_samples
        )

        train_acc = (
            100.0
            * total_correct
            / total_samples
        )

        # ====================================================
        # Validation
        # ====================================================

        if (
            epoch % EVAL_FREQ == 0
            or
            epoch == EPOCHS
        ):

            (
                val_acc,
                val_mean_class_acc
            ) = evaluate_adapter(
                adapter_model=adapter_model,
                image_features=val_features,
                labels=val_labels,
                text_features=text_features,
                logit_scale=logit_scale
            )

            # ------------------------------------------------
            # Save best
            # ------------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = (
                    val_acc
                )

                best_epoch = (
                    epoch
                )

                best_adapter_state = (
                    copy.deepcopy(
                        adapter_model.adapter.state_dict()
                    )
                )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"Epoch [{epoch:03d}/{EPOCHS}] "
                f"| Train Loss: "
                f"{train_loss:.4f} "
                f"| Train Acc: "
                f"{train_acc:.2f}% "
                f"| Val Acc: "
                f"{val_acc:.2f}% "
                f"| LR: "
                f"{current_lr:.6f}"
            )

        else:

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"Epoch [{epoch:03d}/{EPOCHS}] "
                f"| Train Loss: "
                f"{train_loss:.4f} "
                f"| Train Acc: "
                f"{train_acc:.2f}% "
                f"| LR: "
                f"{current_lr:.6f}"
            )

    # ========================================================
    # Restore best Adapter
    # ========================================================

    adapter_model.adapter.load_state_dict(
        best_adapter_state
    )

    return (
        adapter_model,
        best_val_acc,
        best_epoch
    )


# ============================================================
# 14. Save LoRA/Adapter checkpoint
# ============================================================

def save_adapter_checkpoint(
    adapter_model,
    best_epoch
):

    os.makedirs(
        "models",
        exist_ok=True
    )

    torch.save(
        {
            "epoch": best_epoch,
            "feature_dim":
                adapter_model.adapter.feature_dim,
            "bottleneck_dim":
                adapter_model.adapter.bottleneck_dim,
            "reduction":
                adapter_model.adapter.reduction,
            "ratio":
                adapter_model.adapter.ratio,
            "state_dict":
                adapter_model.adapter.state_dict()
        },
        CHECKPOINT_PATH
    )

    print(
        "\nBest Adapter checkpoint saved to:"
    )

    print(
        CHECKPOINT_PATH
    )


# ============================================================
# 15. Save Results
# ============================================================

def save_results(
    best_val_acc,
    best_epoch,
    final_val_acc,
    val_mean_class_acc,
    test_acc,
    test_mean_class_acc,
    total_params,
    trainable_params,
    train_size,
    val_size,
    test_size,
    feature_dim,
    bottleneck_dim
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write("=" * 70 + "\n")

        f.write(
            "CLIP-Adapter Baseline\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        f.write(
            "Experiment: clip_adapter\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        f.write(
            "Backbone: OpenAI CLIP ViT-B/32\n"
        )

        f.write(
            f"Model Path: {MODEL_PATH}\n"
        )

        f.write(
            "CLIP Image Encoder Frozen: Yes\n"
        )

        f.write(
            "CLIP Text Encoder Frozen: Yes\n"
        )

        # ----------------------------------------------------
        # Adapter
        # ----------------------------------------------------

        f.write(
            "Method: CLIP-Adapter\n"
        )

        f.write(
            "Adaptation Side: Visual Feature\n"
        )

        f.write(
            "Adapter Type: Bottleneck MLP\n"
        )

        f.write(
            f"Feature Dimension: "
            f"{feature_dim}\n"
        )

        f.write(
            f"Bottleneck Dimension: "
            f"{bottleneck_dim}\n"
        )

        f.write(
            f"Reduction: "
            f"{ADAPTER_REDUCTION}\n"
        )

        f.write(
            f"Residual Ratio: "
            f"{RESIDUAL_RATIO}\n"
        )

        f.write(
            "Residual Formula: "
            "(1-ratio)*original + ratio*adapter\n"
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        f.write(
            f"Total Parameters: "
            f"{total_params}\n"
        )

        f.write(
            f"Trainable Parameters: "
            f"{trainable_params}\n"
        )

        f.write(
            f"Trainable Ratio: "
            f"{100.0 * trainable_params / total_params:.4f}%\n"
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
        # Prompt
        # ----------------------------------------------------

        f.write(
            f"Prompt Template: "
            f'"{PROMPT_TEMPLATE}"\n'
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        f.write(
            f"Best Validation Accuracy: "
            f"{best_val_acc:.2f}%\n"
        )

        f.write(
            f"Best Epoch: "
            f"{best_epoch}\n"
        )

        f.write(
            f"Final Validation Accuracy: "
            f"{final_val_acc:.2f}%\n"
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
        f"{RESULT_FILE}"
    )


# ============================================================
# 16. Main
# ============================================================

def main():

    set_seed(SEED)

    print(
        "=" * 70
    )

    print(
        "CLIP-Adapter Few-shot Baseline"
    )

    print(
        "=" * 70
    )

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
        "Bottleneck Adapter"
    )

    print(
        "      ↓"
    )

    print(
        "Residual Feature Blending"
    )

    print(
        "      ↓"
    )

    print(
        "Text Feature"
    )

    print(
        "      ↓"
    )

    print(
        "Cosine Similarity"
    )

    print(
        "      ↓"
    )

    print(
        "Classification"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # 1. Load CLIP
    # ========================================================

    clip_model, preprocess = (
        load_clip_model()
    )


    # ========================================================
    # 2. Load Class Names
    # ========================================================

    class_names = (
        load_class_names()
    )

    print(
        "\nClass examples:"
    )

    for i in range(
        min(10, len(class_names))
    ):

        print(
            f"  {i}: {class_names[i]}"
        )


    # ========================================================
    # 3. Load Dataset
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets(
        preprocess
    )


    # ========================================================
    # 4. Build Text Features
    # ========================================================

    text_features = (
        build_text_features(
            clip_model,
            class_names
        )
    )


    # ========================================================
    # 5. Determine feature dimension
    # ========================================================

    # 对 ViT-B/32 应为 512
    feature_dim = (
        clip_model.visual.output_dim
    )

    print(
        f"\nCLIP Image Feature Dimension: "
        f"{feature_dim}"
    )


    # ========================================================
    # 6. Extract Frozen Image Features
    #
    # 因为 CLIP Image Encoder 完全冻结，
    # 我们只需要提取一次。
    # ========================================================

    train_features, train_labels = (
        extract_clip_image_features(
            clip_model,
            train_dataset,
            "Train"
        )
    )

    val_features, val_labels = (
        extract_clip_image_features(
            clip_model,
            val_dataset,
            "Validation"
        )
    )

    test_features, test_labels = (
        extract_clip_image_features(
            clip_model,
            test_dataset,
            "Test"
        )
    )


    # ========================================================
    # 7. Build Adapter Model
    # ========================================================

    adapter_model = (
        CLIPAdapterModel(
            clip_model=clip_model,
            feature_dim=feature_dim,
            reduction=ADAPTER_REDUCTION,
            ratio=RESIDUAL_RATIO
        )
    ).to(DEVICE)


    # ========================================================
    # 8. Trainable Parameters
    # ========================================================

    trainable_parameters = [
        p
        for p in adapter_model.parameters()
        if p.requires_grad
    ]

    trainable_params = sum(
        p.numel()
        for p in trainable_parameters
    )

    total_params = sum(
        p.numel()
        for p in adapter_model.parameters()
    )

    bottleneck_dim = (
        adapter_model.adapter.bottleneck_dim
    )

    print(
        "\nAdapter Parameters:"
    )

    print(
        f"Feature dim: "
        f"{feature_dim}"
    )

    print(
        f"Bottleneck dim: "
        f"{bottleneck_dim}"
    )

    print(
        f"Total params: "
        f"{total_params:,}"
    )

    print(
        f"Trainable params: "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable ratio: "
        f"{100.0 * trainable_params / total_params:.4f}%"
    )


    # ========================================================
    # 9. Logit scale
    #
    # CLIP logit_scale 冻结
    # ========================================================

    logit_scale = (
        clip_model.logit_scale
        .exp()
        .float()
        .detach()
    )


    # ========================================================
    # 10. Train Adapter
    # ========================================================

    (
        adapter_model,
        best_val_acc,
        best_epoch
    ) = train_adapter(
        adapter_model=adapter_model,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        text_features=text_features,
        logit_scale=logit_scale
    )


    # ========================================================
    # 11. Final Validation
    # ========================================================

    (
        final_val_acc,
        final_val_mean_class_acc
    ) = evaluate_adapter(
        adapter_model=adapter_model,
        image_features=val_features,
        labels=val_labels,
        text_features=text_features,
        logit_scale=logit_scale
    )


    # ========================================================
    # 12. Test
    # ========================================================

    print(
        "\nEvaluating Test Set..."
    )

    (
        test_acc,
        test_mean_class_acc
    ) = evaluate_adapter(
        adapter_model=adapter_model,
        image_features=test_features,
        labels=test_labels,
        text_features=text_features,
        logit_scale=logit_scale
    )


    # ========================================================
    # 13. Save Checkpoint
    # ========================================================

    save_adapter_checkpoint(
        adapter_model=adapter_model,
        best_epoch=best_epoch
    )


    # ========================================================
    # 14. Save Results
    # ========================================================

    save_results(
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        final_val_acc=final_val_acc,
        val_mean_class_acc=final_val_mean_class_acc,
        test_acc=test_acc,
        test_mean_class_acc=test_mean_class_acc,
        total_params=total_params,
        trainable_params=trainable_params,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        feature_dim=feature_dim,
        bottleneck_dim=bottleneck_dim
    )


    # ========================================================
    # 15. Final Output
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CLIP-Adapter Experiment Finished"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Final Validation Accuracy: "
        f"{final_val_acc:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    print(
        f"Test Mean Class Accuracy: "
        f"{test_mean_class_acc:.2f}%"
    )

    print(
        f"Trainable Parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable Ratio: "
        f"{100.0 * trainable_params / total_params:.4f}%"
    )

    print(
        "=" * 70
    )


# ============================================================
# 17. Entry
# ============================================================

if __name__ == "__main__":

    main()