import os
import json
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import clip
from tqdm import tqdm


# ============================================================
# 1. 配置参数
# ============================================================

DATA_DIR = "mini-imagenet"

IMAGENET_CLASS_INDEX = os.path.join(
    DATA_DIR,
    "imagenet_class_index.json"
)

MODEL_PATH = os.path.join(
    "models",
    "vit-b-32.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100
K_SHOT = 20

# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 0

# ------------------------------------------------------------
# Tip-Adapter
# ------------------------------------------------------------

ALPHA = 1.0
BETA = 5.0

SEED = 42

# ------------------------------------------------------------
# Zero-shot Prompt
# ------------------------------------------------------------

PROMPT_TEMPLATE = "a photo of a {}"


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
# 3. 加载 OpenAI CLIP
# ============================================================

def load_clip_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"找不到 CLIP 模型：{MODEL_PATH}"
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
        "Model: CLIP ViT-B/32"
    )

    print(
        f"CLIP dtype: {model.dtype}"
    )

    print(
        "Backbone: Frozen"
    )

    print("=" * 70)

    return model, preprocess


# ============================================================
# 4. ImageNet 类别映射
# ============================================================

def load_imagenet_mapping():

    if not os.path.exists(
        IMAGENET_CLASS_INDEX
    ):

        raise FileNotFoundError(
            f"找不到：{IMAGENET_CLASS_INDEX}"
        )

    with open(
        IMAGENET_CLASS_INDEX,
        "r",
        encoding="utf-8"
    ) as f:

        class_index = json.load(f)

    synset_to_name = {}

    for _, value in class_index.items():

        if (
            not isinstance(value, list)
            or len(value) < 2
        ):
            continue

        wnid = value[0]

        class_name = (
            value[1]
            .replace("_", " ")
            .replace("-", " ")
        )

        synset_to_name[wnid] = class_name

    return synset_to_name


# ============================================================
# 5. 加载数据集
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
    # 注意：
    #
    # 这里直接把 CLIP preprocess 交给 ImageFolder。
    #
    # 所以 dataset[index] 返回的 image 已经是 Tensor，
    # 后面不能再次执行 preprocess(image)。
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
    # 类别检查
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
    ), "类别数量不是 100！"

    # --------------------------------------------------------
    # 20-shot 检查
    # --------------------------------------------------------

    expected_train_size = (
        NUM_CLASSES * K_SHOT
    )

    if len(train_dataset) != expected_train_size:

        raise ValueError(
            f"Train Size={len(train_dataset)}，"
            f"预期应为 {expected_train_size}。"
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
# 6. DataLoader
# ============================================================

def build_dataloader(dataset):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=False
    )


# ============================================================
# 7. 提取 CLIP Image Features
# ============================================================

@torch.no_grad()
def extract_image_features(
    model,
    dataset,
    description
):

    loader = build_dataloader(
        dataset
    )

    model.eval()

    all_features = []
    all_labels = []

    print(
        f"\n正在提取 {description} Image Features..."
    )

    for images, labels in tqdm(
        loader,
        desc=f"{description} Features"
    ):

        # ----------------------------------------------------
        # 这里的 images 已经经过 CLIP preprocess
        # 所以直接送入模型
        # ----------------------------------------------------

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Image Encoder
        # ----------------------------------------------------

        features = model.encode_image(
            images
        )

        # ----------------------------------------------------
        # 转 FP32
        # ----------------------------------------------------

        features = features.float()

        # ----------------------------------------------------
        # L2 Normalize
        # ----------------------------------------------------

        features = (
            features
            / (
                features.norm(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )
        )

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
        f"{description} Feature Shape: "
        f"{tuple(features.shape)}"
    )

    return features, labels


# ============================================================
# 8. 构造 Text Features
# ============================================================

@torch.no_grad()
def build_text_features(
    model,
    class_names
):

    prompts = []

    for class_name in class_names:

        prompts.append(
            PROMPT_TEMPLATE.format(
                class_name
            )
        )

    print(
        "\nZero-shot Prompt 示例："
    )

    for prompt in prompts[:10]:

        print(
            f"  {prompt}"
        )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    text_tokens = clip.tokenize(
        prompts
    ).to(DEVICE)

    # --------------------------------------------------------
    # Text Encoder
    # --------------------------------------------------------

    text_features = model.encode_text(
        text_tokens
    )

    text_features = (
        text_features.float()
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    text_features = (
        text_features
        / (
            text_features.norm(
                dim=-1,
                keepdim=True
            )
            + 1e-8
        )
    )

    print(
        f"Text Feature Shape: "
        f"{tuple(text_features.shape)}"
    )

    return text_features


# ============================================================
# 9. 构建 Tip-Adapter Cache
# ============================================================

def build_cache_model(
    train_features,
    train_labels
):

    print(
        "\n正在构建 Tip-Adapter Cache..."
    )

    # --------------------------------------------------------
    # Cache Keys
    #
    # 每张 few-shot 图像对应一个 CLIP feature
    # --------------------------------------------------------

    cache_keys = (
        train_features.clone()
    )

    # --------------------------------------------------------
    # Cache Values
    #
    # One-hot label
    # --------------------------------------------------------

    cache_values = torch.zeros(
        train_labels.shape[0],
        NUM_CLASSES,
        dtype=torch.float32
    )

    cache_values[
        torch.arange(
            train_labels.shape[0]
        ),
        train_labels
    ] = 1.0

    print(
        f"Cache Keys Shape  : "
        f"{tuple(cache_keys.shape)}"
    )

    print(
        f"Cache Values Shape: "
        f"{tuple(cache_values.shape)}"
    )

    return (
        cache_keys,
        cache_values
    )


# ============================================================
# 10. CLIP Zero-shot Logits
# ============================================================

def compute_clip_logits(
    image_features,
    text_features,
    logit_scale
):

    return (
        logit_scale
        * (
            image_features
            @ text_features.T
        )
    )


# ============================================================
# 11. Tip-Adapter Cache Logits
# ============================================================

def compute_cache_logits(
    image_features,
    cache_keys,
    cache_values,
    beta
):

    # --------------------------------------------------------
    # Image-Cache similarity
    # --------------------------------------------------------

    affinity = (
        image_features
        @ cache_keys.T
    )

    # --------------------------------------------------------
    # Tip-Adapter affinity
    #
    # exp(-beta * (1 - similarity))
    # --------------------------------------------------------

    affinity = torch.exp(
        -beta
        * (
            1.0
            - affinity
        )
    )

    # --------------------------------------------------------
    # Key-Value retrieval
    # --------------------------------------------------------

    cache_logits = (
        affinity
        @ cache_values.to(DEVICE)
    )

    return cache_logits


# ============================================================
# 12. Tip-Adapter Evaluation
# ============================================================

@torch.no_grad()
def evaluate_tip_adapter(
    model,
    image_features,
    labels,
    text_features,
    cache_keys,
    cache_values,
    alpha,
    beta,
    description
):

    print(
        f"\n开始 {description} Tip-Adapter Evaluation..."
    )

    # --------------------------------------------------------
    # CLIP logit scale
    # --------------------------------------------------------

    logit_scale = (
        model.logit_scale
        .exp()
        .float()
        .detach()
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    image_features = image_features.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    # --------------------------------------------------------
    # CLIP logits
    # --------------------------------------------------------

    clip_logits = (
        compute_clip_logits(
            image_features,
            text_features,
            logit_scale
        )
    )

    # --------------------------------------------------------
    # Cache logits
    # --------------------------------------------------------

    cache_logits = (
        compute_cache_logits(
            image_features,
            cache_keys,
            cache_values,
            beta
        )
    )

    # --------------------------------------------------------
    # Tip-Adapter logits
    # --------------------------------------------------------

    tip_logits = (
        clip_logits
        + alpha * cache_logits
    )

    predictions = (
        tip_logits.argmax(
            dim=1
        )
    )

    # ========================================================
    # Overall Accuracy
    # ========================================================

    accuracy = (
        100.0
        * (
            predictions == labels
        ).sum().item()
        /
        len(labels)
    )

    # ========================================================
    # Mean Class Accuracy
    # ========================================================

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
# 13. 保存结果
# ============================================================

def save_results(
    val_acc,
    val_mean_class_acc,
    test_acc,
    test_mean_class_acc,
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
            "CLIP Tip-Adapter Baseline\n"
        )
        f.write(
            "=" * 70 + "\n"
        )

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        f.write(
            "Experiment: clip_tip_adapter\n"
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
        # Tip-Adapter
        # ----------------------------------------------------

        f.write(
            "Method: Tip-Adapter\n"
        )

        f.write(
            "Variant: Training-free\n"
        )

        f.write(
            "Cache Type: Image Feature Key-Value Cache\n"
        )

        f.write(
            f"Alpha: {ALPHA}\n"
        )

        f.write(
            f"Beta: {BETA}\n"
        )

        f.write(
            "Backpropagation: No\n"
        )

        f.write(
            "Parameter Fine-tuning: No\n"
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        f.write(
            f"Prompt Template: "
            f'"{PROMPT_TEMPLATE}"\n'
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
        # Results
        # ----------------------------------------------------

        f.write(
            f"Validation Accuracy: "
            f"{val_acc:.2f}%\n"
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
# 14. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("CLIP Tip-Adapter Few-shot Adaptation")
    print("=" * 70)

    print(
        "Frozen CLIP"
    )

    print(
        "      ↓"
    )

    print(
        "20-shot Image Feature Cache"
    )

    print(
        "      ↓"
    )

    print(
        "Feature Retrieval"
    )

    print(
        "      ↓"
    )

    print(
        "CLIP Logits + Cache Logits"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"K-shot: {K_SHOT}"
    )

    print(
        f"Alpha : {ALPHA}"
    )

    print(
        f"Beta  : {BETA}"
    )


    # ========================================================
    # 1. CLIP
    # ========================================================

    model, preprocess = (
        load_clip_model()
    )


    # ========================================================
    # 2. ImageNet Mapping
    # ========================================================

    synset_to_name = (
        load_imagenet_mapping()
    )


    # ========================================================
    # 3. Dataset
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets(
        preprocess
    )


    # ========================================================
    # 4. 类别名称
    # ========================================================

    class_names = []

    for synset in train_dataset.classes:

        if synset not in synset_to_name:

            raise KeyError(
                f"{synset} 不存在于 "
                f"imagenet_class_index.json"
            )

        class_names.append(
            synset_to_name[synset]
        )

    print(
        "\n类别名称示例："
    )

    for name in class_names[:10]:

        print(
            f"  {name}"
        )


    # ========================================================
    # 5. Text Features
    # ========================================================

    text_features = (
        build_text_features(
            model,
            class_names
        )
    )


    # ========================================================
    # 6. Image Features
    #
    # Train:
    # 用来建立 Cache
    #
    # Val/Test:
    # 仅用于评估
    # ========================================================

    train_features, train_labels = (
        extract_image_features(
            model,
            train_dataset,
            "Train"
        )
    )

    val_features, val_labels = (
        extract_image_features(
            model,
            val_dataset,
            "Validation"
        )
    )

    test_features, test_labels = (
        extract_image_features(
            model,
            test_dataset,
            "Test"
        )
    )


    # ========================================================
    # 7. Cache
    # ========================================================

    (
        cache_keys,
        cache_values
    ) = build_cache_model(
        train_features,
        train_labels
    )

    cache_keys = (
        cache_keys.to(
            DEVICE
        )
    )


    # ========================================================
    # 8. Validation
    #
    # 固定 ALPHA / BETA
    # 不在这里搜索超参数
    # ========================================================

    (
        val_acc,
        val_mean_class_acc
    ) = evaluate_tip_adapter(
        model=model,
        image_features=val_features,
        labels=val_labels,
        text_features=text_features,
        cache_keys=cache_keys,
        cache_values=cache_values,
        alpha=ALPHA,
        beta=BETA,
        description="Validation"
    )


    # ========================================================
    # 9. Test
    # ========================================================

    (
        test_acc,
        test_mean_class_acc
    ) = evaluate_tip_adapter(
        model=model,
        image_features=test_features,
        labels=test_labels,
        text_features=text_features,
        cache_keys=cache_keys,
        cache_values=cache_values,
        alpha=ALPHA,
        beta=BETA,
        description="Test"
    )


    # ========================================================
    # 10. 保存结果
    # ========================================================

    save_results(
        val_acc=val_acc,
        val_mean_class_acc=val_mean_class_acc,
        test_acc=test_acc,
        test_mean_class_acc=test_mean_class_acc,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset)
    )


    # ========================================================
    # 11. 最终输出
    # ========================================================

    print("\n" + "=" * 70)
    print("CLIP Tip-Adapter 实验结束")
    print("=" * 70)

    print(
        f"Validation Accuracy: "
        f"{val_acc:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    print(
        f"Test Mean Class Accuracy: "
        f"{test_mean_class_acc:.2f}%"
    )

    print("=" * 70)


# ============================================================
# 15. 程序入口
# ============================================================

if __name__ == "__main__":
    main()