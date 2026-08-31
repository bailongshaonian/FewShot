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

# ImageNet 类别映射
IMAGENET_CLASS_INDEX = os.path.join(
    DATA_DIR,
    "imagenet_class_index.json"
)

# OpenAI CLIP 官方 checkpoint
MODEL_PATH = os.path.join(
    "models",
    "vit-b-32.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100

BATCH_SIZE = 32

# Windows 下使用 0，保证稳定性
NUM_WORKERS = 0

SEED = 42

# Zero-shot Prompt
PROMPT_TEMPLATE = "a photo of a {}"


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


set_seed(SEED)


# ============================================================
# 3. 加载 ImageNet 类别映射
# ============================================================

def load_class_mapping():

    if not os.path.exists(
        IMAGENET_CLASS_INDEX
    ):
        raise FileNotFoundError(
            f"找不到 ImageNet 类别映射文件：\n"
            f"{IMAGENET_CLASS_INDEX}"
        )

    print(
        "\n正在加载 ImageNet 类别映射..."
    )

    with open(
        IMAGENET_CLASS_INDEX,
        "r",
        encoding="utf-8"
    ) as f:

        class_index = json.load(f)

    # --------------------------------------------------------
    # JSON:
    #
    # {
    #   "0": ["n01440764", "tench"],
    #   "1": ["n01443537", "goldfish"],
    #   ...
    # }
    #
    # 转换为：
    #
    # {
    #   "n01440764": "tench",
    #   "n01443537": "goldfish",
    #   ...
    # }
    # --------------------------------------------------------

    synset_to_name = {}

    for _, value in class_index.items():

        if (
            not isinstance(value, list)
            or len(value) < 2
        ):
            continue

        synset = value[0]
        class_name = value[1]

        synset_to_name[synset] = class_name

    print(
        f"共加载 {len(synset_to_name)} 个 ImageNet 类别。"
    )

    return synset_to_name


# ============================================================
# 4. 加载 OpenAI CLIP
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

    print(
        "CLIP Model: ViT-B/32"
    )

    print("=" * 70)

    return model, preprocess


# ============================================================
# 5. 加载 Validation / Test
# ============================================================

def load_datasets():

    val_dir = os.path.join(
        DATA_DIR,
        "val"
    )

    test_dir = os.path.join(
        DATA_DIR,
        "test"
    )

    # 不设置 transform
    # 由 CLIP preprocess 完成图像预处理
    val_dataset = datasets.ImageFolder(
        val_dir
    )

    test_dataset = datasets.ImageFolder(
        test_dir
    )

    # --------------------------------------------------------
    # 类别检查
    # --------------------------------------------------------

    assert (
        val_dataset.classes
        == test_dataset.classes
    ), "Validation 和 Test 类别顺序不一致！"

    assert (
        len(val_dataset.classes)
        == NUM_CLASSES
    ), (
        f"类别数量不是 {NUM_CLASSES}，"
        f"实际为 {len(val_dataset.classes)}"
    )

    print("\n数据集信息：")

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test images       : "
        f"{len(test_dataset)}"
    )

    print(
        f"Classes           : "
        f"{len(val_dataset.classes)}"
    )

    return val_dataset, test_dataset


# ============================================================
# 6. 将 Synset ID 映射到人类可读类别名称
# ============================================================

def map_class_names(
    class_names,
    synset_to_name
):

    mapped_names = []

    print(
        "\nImageNet 类别映射示例："
    )

    for index, synset in enumerate(
        class_names
    ):

        if synset not in synset_to_name:

            raise KeyError(
                f"类别 {synset} "
                f"在 imagenet_class_index.json 中不存在！"
            )

        readable_name = (
            synset_to_name[synset]
        )

        mapped_names.append(
            readable_name
        )

        if index < 10:

            print(
                f"  {synset:12s}"
                f" -> "
                f"{readable_name}"
            )

    return mapped_names


# ============================================================
# 7. 构造 Zero-shot Prompt
# ============================================================

def build_prompts(
    readable_class_names
):

    prompts = []

    for class_name in readable_class_names:

        # ImageNet JSON 中部分类别名称可能包含逗号
        # 这里直接保留类别名称的语义内容

        clean_name = (
            class_name
            .replace("_", " ")
        )

        prompt = (
            PROMPT_TEMPLATE.format(
                clean_name
            )
        )

        prompts.append(
            prompt
        )

    return prompts


# ============================================================
# 8. 构造 Text Embeddings
# ============================================================

@torch.no_grad()
def build_text_embeddings(
    model,
    prompts
):

    print(
        "\n正在构造 Zero-shot Text Embeddings..."
    )

    print("\n最终使用的 Prompt：")

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
    # CLIP Text Encoder
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
        f"\nText embedding shape: "
        f"{tuple(text_features.shape)}"
    )

    return text_features


# ============================================================
# 9. DataLoader
# ============================================================

def clip_collate_fn(batch):

    images, labels = zip(*batch)

    return (
        list(images),
        torch.tensor(
            labels,
            dtype=torch.long
        )
    )


def build_dataloader(dataset):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=False,
        collate_fn=clip_collate_fn
    )


# ============================================================
# 10. Zero-shot Evaluation
# ============================================================

@torch.no_grad()
def evaluate_zero_shot(
    model,
    preprocess,
    dataset,
    text_features,
    description
):

    print(
        f"\n开始 {description} Zero-shot Evaluation..."
    )

    loader = build_dataloader(
        dataset
    )

    model.eval()

    correct = 0
    total = 0

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for images, labels in tqdm(
        loader,
        desc=f"{description} Zero-shot"
    ):

        # ----------------------------------------------------
        # CLIP 官方预处理
        # ----------------------------------------------------

        image_inputs = torch.stack([
            preprocess(image)
            for image in images
        ]).to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Image Encoder
        # ----------------------------------------------------

        image_features = model.encode_image(
            image_inputs
        )

        image_features = (
            image_features.float()
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image_features = (
            image_features
            / (
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        # ----------------------------------------------------
        # Cosine Similarity
        # ----------------------------------------------------

        similarity = (
            image_features
            @ text_features.T
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = similarity.argmax(
            dim=1
        )

        labels = labels.to(
            DEVICE
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += (
            labels.size(0)
        )

        # ----------------------------------------------------
        # Per-class statistics
        # ----------------------------------------------------

        for class_id in range(
            NUM_CLASSES
        ):

            mask = (
                labels == class_id
            )

            if mask.any():

                class_total[class_id] += (
                    mask.sum().item()
                )

                class_correct[class_id] += (
                    (
                        predictions[mask]
                        == labels[mask]
                    )
                    .sum()
                    .item()
                )

    # ========================================================
    # Overall Accuracy
    # ========================================================

    accuracy = (
        100.0
        * correct
        / total
    )

    # ========================================================
    # Mean Class Accuracy
    # ========================================================

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
# 11. 保存结果
# ============================================================

def save_results(
    val_acc,
    val_mean_class_acc,
    test_acc,
    test_mean_class_acc,
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
            "Foundation Model Zero-shot Baseline\n"
        )
        f.write("=" * 70 + "\n")

        # 实验身份
        f.write(
            "Experiment: clip_zero_shot\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        # Model
        f.write(
            "Model: OpenAI CLIP ViT-B/32\n"
        )

        f.write(
            f"Model Path: {MODEL_PATH}\n"
        )

        f.write(
            "Model Type: "
            "Vision-Language Foundation Model\n"
        )

        f.write(
            "Training Mode: Zero-shot Inference\n"
        )

        f.write(
            "Backbone Frozen: Yes\n"
        )

        f.write(
            "Fine-tuning: No\n"
        )

        # Prompt
        f.write(
            f"Prompt Template: "
            f'"{PROMPT_TEMPLATE}"\n'
        )

        # Zero-shot
        f.write(
            "Training Samples Used: 0\n"
        )

        f.write(
            "Support Set Used: No\n"
        )

        f.write(
            "Validation Used for Model Selection: No\n"
        )

        f.write(
            "Test Set Used for Training: No\n"
        )

        # Data
        f.write(
            f"Num Classes: {NUM_CLASSES}\n"
        )

        f.write(
            f"Validation Size: {val_size}\n"
        )

        f.write(
            f"Test Size: {test_size}\n"
        )

        # Results
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
# 12. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("CLIP Zero-shot Baseline")
    print("=" * 70)

    print(
        "Mini-ImageNet Synset ID"
    )

    print(
        "        ↓"
    )

    print(
        "ImageNet Class Name Mapping"
    )

    print(
        "        ↓"
    )

    print(
        "Natural Language Prompt"
    )

    print(
        "        ↓"
    )

    print(
        "CLIP Text Encoder"
    )

    print(
        "        ↕"
    )

    print(
        "Cosine Similarity"
    )

    print(
        "        ↕"
    )

    print(
        "CLIP Image Encoder"
    )

    print("=" * 70)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Model  : {MODEL_PATH}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print(
        f"Prompt : {PROMPT_TEMPLATE}"
    )


    # ========================================================
    # 1. 加载类别映射
    # ========================================================

    synset_to_name = (
        load_class_mapping()
    )


    # ========================================================
    # 2. 加载 CLIP
    # ========================================================

    model, preprocess = (
        load_clip_model()
    )


    # ========================================================
    # 3. 加载数据集
    # ========================================================

    val_dataset, test_dataset = (
        load_datasets()
    )


    # ========================================================
    # 4. 映射类别名称
    # ========================================================

    readable_class_names = (
        map_class_names(
            val_dataset.classes,
            synset_to_name
        )
    )


    # ========================================================
    # 5. 构造 Prompt
    # ========================================================

    prompts = build_prompts(
        readable_class_names
    )


    # ========================================================
    # 6. Text Embeddings
    # ========================================================

    text_features = (
        build_text_embeddings(
            model=model,
            prompts=prompts
        )
    )


    # ========================================================
    # 7. Validation
    #
    # 仅记录结果，不用于参数选择。
    # ========================================================

    (
        val_acc,
        val_mean_class_acc
    ) = evaluate_zero_shot(
        model=model,
        preprocess=preprocess,
        dataset=val_dataset,
        text_features=text_features,
        description="Validation"
    )


    # ========================================================
    # 8. Test
    # ========================================================

    (
        test_acc,
        test_mean_class_acc
    ) = evaluate_zero_shot(
        model=model,
        preprocess=preprocess,
        dataset=test_dataset,
        text_features=text_features,
        description="Test"
    )


    # ========================================================
    # 9. 保存结果
    # ========================================================

    save_results(
        val_acc=val_acc,
        val_mean_class_acc=val_mean_class_acc,
        test_acc=test_acc,
        test_mean_class_acc=test_mean_class_acc,
        val_size=len(val_dataset),
        test_size=len(test_dataset)
    )


    # ========================================================
    # 10. 最终输出
    # ========================================================

    print("\n" + "=" * 70)
    print("CLIP Zero-shot 实验结束")
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
# 13. 程序入口
# ============================================================

if __name__ == "__main__":
    main()