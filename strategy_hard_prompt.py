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

# 你生成的 WordNet 类别描述
CLASS_DESCRIPTIONS = os.path.join(
    DATA_DIR,
    "class_descriptions.json"
)

# OpenAI CLIP
MODEL_PATH = os.path.join(
    "models",
    "vit-b-32.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100

BATCH_SIZE = 32

# Windows 下保持稳定
NUM_WORKERS = 0

SEED = 42

# ------------------------------------------------------------
# Hard Prompt
# ------------------------------------------------------------

PROMPT_TEMPLATE = (
    "a photo of a {class_name}, {description}"
)


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

    print(
        "Model: CLIP ViT-B/32"
    )

    print("=" * 70)

    return model, preprocess


# ============================================================
# 4. 加载类别描述
# ============================================================

def load_class_descriptions():

    if not os.path.exists(
        CLASS_DESCRIPTIONS
    ):

        raise FileNotFoundError(
            f"找不到类别描述文件：\n"
            f"{CLASS_DESCRIPTIONS}"
        )

    print(
        "\n正在加载 WordNet 类别描述..."
    )

    with open(
        CLASS_DESCRIPTIONS,
        "r",
        encoding="utf-8"
    ) as f:

        descriptions = json.load(f)

    print(
        f"加载了 {len(descriptions)} 个类别描述。"
    )

    return descriptions


# ============================================================
# 5. 加载 Dataset
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

    # --------------------------------------------------------
    # 不使用 transform
    #
    # CLIP preprocess 在后面统一处理
    # --------------------------------------------------------

    val_dataset = datasets.ImageFolder(
        val_dir
    )

    test_dataset = datasets.ImageFolder(
        test_dir
    )

    # --------------------------------------------------------
    # 检查类别
    # --------------------------------------------------------

    assert (
        val_dataset.classes
        == test_dataset.classes
    ), "Validation 与 Test 类别顺序不一致！"

    assert (
        len(val_dataset.classes)
        == NUM_CLASSES
    ), (
        f"类别数量不是 {NUM_CLASSES}"
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
        f"Classes            : "
        f"{len(val_dataset.classes)}"
    )

    return (
        val_dataset,
        test_dataset
    )


# ============================================================
# 6. 构造 Hard Prompts
# ============================================================

def build_hard_prompts(
    class_names,
    descriptions
):

    prompts = []

    print(
        "\n正在构造 Knowledge-enhanced Hard Prompts..."
    )

    for class_name in class_names:

        # ----------------------------------------------------
        # class_name 本身应该是 ImageNet synset，例如：
        #
        # n01532829
        #
        # 但 class_descriptions.json 的 key 是：
        #
        # 英文类别名
        # ----------------------------------------------------

        raise_error = False

        # ----------------------------------------------------
        # 找到对应描述
        # ----------------------------------------------------

        # 当前我们需要从 ImageNet class index
        # 完成 synset -> readable name -> description
        #
        # 因此这里只先保留接口。
        # 实际 mapping 在下面 build_prompts_with_mapping()
        # 中完成。
        # ----------------------------------------------------

    return prompts


# ============================================================
# 7. 加载 ImageNet synset → class name
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
# 8. 构造最终 Hard Prompts
# ============================================================

def build_prompts(
    class_names,
    synset_to_name,
    descriptions
):

    prompts = []

    print(
        "\n正在构造最终 Hard Prompts..."
    )

    for synset in class_names:

        # ----------------------------------------------------
        # synset -> readable class name
        # ----------------------------------------------------

        if synset not in synset_to_name:

            raise KeyError(
                f"{synset} 不存在于 "
                f"imagenet_class_index.json"
            )

        class_name = (
            synset_to_name[synset]
        )

        # ----------------------------------------------------
        # readable name -> description
        # ----------------------------------------------------

        if class_name not in descriptions:

            raise KeyError(
                f"{class_name} 不存在于 "
                f"class_descriptions.json"
            )

        info = descriptions[
            class_name
        ]

        # ----------------------------------------------------
        # 兼容新的 JSON 结构：
        #
        # {
        #   "wnid": "...",
        #   "synset": "...",
        #   "definition": "...",
        #   "description": "which is ..."
        # }
        # ----------------------------------------------------

        if isinstance(info, dict):

            description = info.get(
                "description",
                ""
            )

        else:

            # 兼容你最开始生成的旧 JSON 格式
            description = info

        if not description:

            raise ValueError(
                f"{class_name} 的 description 为空！"
            )

        prompt = PROMPT_TEMPLATE.format(
            class_name=class_name,
            description=description
        )

        prompts.append(
            prompt
        )

    # --------------------------------------------------------
    # 检查数量
    # --------------------------------------------------------

    if len(prompts) != NUM_CLASSES:

        raise ValueError(
            f"Prompt 数量为 {len(prompts)}，"
            f"而类别数量为 {NUM_CLASSES}。"
        )

    # --------------------------------------------------------
    # 预览
    # --------------------------------------------------------

    print(
        "\n>>> Hard Prompt 预览："
    )

    for i, prompt in enumerate(
        prompts[:10]
    ):

        print(
            f"{i:03d}: {prompt}"
        )

    return prompts


# ============================================================
# 9. 构造 Text Embeddings
# ============================================================

@torch.no_grad()
def build_text_embeddings(
    model,
    prompts
):

    print(
        "\n正在编码 Hard Prompt..."
    )

    text_tokens = clip.tokenize(
        prompts
    ).to(DEVICE)

    text_features = model.encode_text(
        text_tokens
    )

    text_features = (
        text_features.float()
    )

    # --------------------------------------------------------
    # L2 Normalize
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
        f"Text embedding shape: "
        f"{tuple(text_features.shape)}"
    )

    return text_features


# ============================================================
# 10. DataLoader
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
# 11. Zero-shot Evaluation
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
        f"\n开始 {description} Hard Prompt Evaluation..."
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
        desc=f"{description} Hard Prompt"
    ):

        # ----------------------------------------------------
        # CLIP 官方图像预处理
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
        # Similarity
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
# 12. 保存结果
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
            "Knowledge-Enhanced Hard Prompt Zero-shot\n"
        )
        f.write("=" * 70 + "\n")

        # ----------------------------------------------------
        # 实验信息
        # ----------------------------------------------------

        f.write(
            "Experiment: "
            "clip_hard_prompt_zero_shot\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        f.write(
            "Model: OpenAI CLIP ViT-B/32\n"
        )

        f.write(
            f"Model Path: {MODEL_PATH}\n"
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

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        f.write(
            "Prompt Type: "
            "Knowledge-Enhanced Hard Prompt\n"
        )

        f.write(
            f"Prompt Template: "
            f'"{PROMPT_TEMPLATE}"\n'
        )

        f.write(
            "External Knowledge: "
            "WordNet Definition\n"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        f.write(
            "Training Samples Used: 0\n"
        )

        f.write(
            "Support Set Used: No\n"
        )

        f.write(
            "Parameter Optimization: No\n"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        f.write(
            f"Num Classes: {NUM_CLASSES}\n"
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
# 13. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Knowledge-Enhanced Hard Prompt Zero-shot")
    print("=" * 70)

    print(
        "ImageNet Synset"
    )

    print(
        "      ↓"
    )

    print(
        "WordNet Semantic Description"
    )

    print(
        "      ↓"
    )

    print(
        "Hard Prompt"
    )

    print(
        "      ↓"
    )

    print(
        "CLIP Text Encoder"
    )

    print(
        "      ↕"
    )

    print(
        "Cosine Similarity"
    )

    print(
        "      ↕"
    )

    print(
        "CLIP Image Encoder"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Prompt: {PROMPT_TEMPLATE}"
    )


    # ========================================================
    # 1. 加载类别映射
    # ========================================================

    synset_to_name = (
        load_imagenet_mapping()
    )


    # ========================================================
    # 2. 加载类别描述
    # ========================================================

    descriptions = (
        load_class_descriptions()
    )


    # ========================================================
    # 3. 加载 CLIP
    # ========================================================

    model, preprocess = (
        load_clip_model()
    )


    # ========================================================
    # 4. 加载数据
    # ========================================================

    (
        val_dataset,
        test_dataset
    ) = load_datasets()


    # ========================================================
    # 5. 构造 Hard Prompt
    # ========================================================

    prompts = build_prompts(
        class_names=val_dataset.classes,
        synset_to_name=synset_to_name,
        descriptions=descriptions
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
    # 不用于训练和参数选择
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
    print(
        "Knowledge-Enhanced Hard Prompt 实验结束"
    )
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
# 14. 程序入口
# ============================================================

if __name__ == "__main__":
    main()