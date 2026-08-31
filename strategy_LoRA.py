import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============================================================
# LoRA 配置
# ============================================================

LORA_RANK = 4

LORA_ALPHA = 8.0

LORA_DROPOUT = 0.0


# ============================================================
# Training 配置
# ============================================================

BATCH_SIZE = 64

EPOCHS = 100

LR = 0.001

WEIGHT_DECAY = 1e-4

EVAL_FREQ = 5

NUM_WORKERS = 0

SEED = 42


# ============================================================
# Prompt
# ============================================================

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
# 3. LoRA Linear
# ============================================================

class LoRALinear(nn.Module):

    """
    LoRA 包装器

    原始：
        y = W x + b

    LoRA：
        y = W x + b + scaling * B(Ax)

    其中：
        W / b : 冻结
        A / B : 可训练

    dtype:
        原始 CLIP 权重通常为 FP16
        LoRA 参数保持 FP32

    device:
        LoRA 参数与原始 Linear 保持相同 device
    """

    def __init__(
        self,
        original_linear,
        rank=4,
        alpha=8.0,
        dropout=0.0
    ):

        super().__init__()

        if not isinstance(
            original_linear,
            nn.Linear
        ):
            raise TypeError(
                "LoRALinear 只能包装 nn.Linear"
            )

        self.in_features = (
            original_linear.in_features
        )

        self.out_features = (
            original_linear.out_features
        )

        self.rank = rank

        self.alpha = alpha

        self.scaling = (
            alpha / rank
        )

        # ----------------------------------------------------
        # 保存原始 Linear
        # ----------------------------------------------------

        self.original = original_linear

        for param in self.original.parameters():

            param.requires_grad = False

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            p=dropout
        )

        # ----------------------------------------------------
        # 关键：
        # LoRA 参数与原始 Linear 保持相同 device
        # ----------------------------------------------------

        device = (
            original_linear.weight.device
        )

        # ----------------------------------------------------
        # LoRA A
        # ----------------------------------------------------

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                dtype=torch.float32,
                device=device
            )
        )

        # ----------------------------------------------------
        # LoRA B
        # ----------------------------------------------------

        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                dtype=torch.float32,
                device=device
            )
        )

        # ----------------------------------------------------
        # 初始化
        #
        # A 随机
        # B = 0
        #
        # 初始 LoRA 输出为 0
        # 因此初始模型与原始 CLIP 完全一致
        # ----------------------------------------------------

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=np.sqrt(5)
        )

        nn.init.zeros_(
            self.lora_B
        )

    def forward(self, x):

        # ====================================================
        # 1. Original CLIP Linear
        # ====================================================

        base_output = (
            self.original(x)
        )

        # ====================================================
        # 2. LoRA branch
        # ====================================================

        x_float = (
            self.dropout(x)
            .float()
        )

        # ----------------------------------------------------
        # 确保 device 一致
        # ----------------------------------------------------

        x_float = x_float.to(
            self.lora_A.device
        )

        # ----------------------------------------------------
        # A
        # ----------------------------------------------------

        lora_output = F.linear(
            x_float,
            self.lora_A
        )

        # ----------------------------------------------------
        # B
        # ----------------------------------------------------

        lora_output = F.linear(
            lora_output,
            self.lora_B
        )

        # ----------------------------------------------------
        # scaling
        # ----------------------------------------------------

        lora_output = (
            lora_output
            * self.scaling
        )

        # ----------------------------------------------------
        # 转回原始 CLIP dtype
        # ----------------------------------------------------

        lora_output = lora_output.to(
            dtype=base_output.dtype,
            device=base_output.device
        )

        # ====================================================
        # 3. Residual
        # ====================================================

        return (
            base_output
            + lora_output
        )


# ============================================================
# 4. 向视觉 Transformer MLP 注入 LoRA
# ============================================================

def inject_lora_into_visual_mlp(
    model,
    rank,
    alpha,
    dropout
):

    print(
        "\n正在向 CLIP Visual Transformer 注入 LoRA..."
    )

    visual_transformer = (
        model.visual.transformer
    )

    resblocks = (
        visual_transformer.resblocks
    )

    replaced_layers = []

    for block_idx, block in enumerate(
        resblocks
    ):

        mlp = block.mlp

        # ----------------------------------------------------
        # MLP c_fc
        # ----------------------------------------------------

        if hasattr(
            mlp,
            "c_fc"
        ):

            if isinstance(
                mlp.c_fc,
                nn.Linear
            ):

                mlp.c_fc = LoRALinear(
                    original_linear=mlp.c_fc,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )

                replaced_layers.append(
                    f"visual.transformer.resblocks."
                    f"{block_idx}.mlp.c_fc"
                )

        # ----------------------------------------------------
        # MLP c_proj
        # ----------------------------------------------------

        if hasattr(
            mlp,
            "c_proj"
        ):

            if isinstance(
                mlp.c_proj,
                nn.Linear
            ):

                mlp.c_proj = LoRALinear(
                    original_linear=mlp.c_proj,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )

                replaced_layers.append(
                    f"visual.transformer.resblocks."
                    f"{block_idx}.mlp.c_proj"
                )

    if len(replaced_layers) == 0:

        raise RuntimeError(
            "没有成功注入任何 LoRA 层！"
        )

    print(
        f"成功注入 "
        f"{len(replaced_layers)} "
        f"个 LoRA Linear Layer。"
    )

    return replaced_layers


# ============================================================
# 5. 冻结 CLIP，仅保留 LoRA 参数
# ============================================================

def freeze_clip_except_lora(
    model
):

    # --------------------------------------------------------
    # 全部冻结
    # --------------------------------------------------------

    for param in model.parameters():

        param.requires_grad = False

    # --------------------------------------------------------
    # 打开 LoRA 参数
    # --------------------------------------------------------

    trainable_names = []

    for name, param in model.named_parameters():

        if (
            "lora_A" in name
            or
            "lora_B" in name
        ):

            param.requires_grad = True

            trainable_names.append(
                name
            )

    if len(trainable_names) == 0:

        raise RuntimeError(
            "没有找到可训练 LoRA 参数！"
        )

    print(
        "\nLoRA 可训练参数："
    )

    for name in trainable_names:

        print(
            f"  {name}"
        )

    return trainable_names


# ============================================================
# 6. 检查 LoRA 参数 device / dtype
# ============================================================

def inspect_lora_parameters(
    model
):

    print(
        "\nLoRA 参数检查："
    )

    found = False

    for name, param in model.named_parameters():

        if (
            "lora_A" in name
            or
            "lora_B" in name
        ):

            found = True

            print(
                f"{name}"
            )

            print(
                f"  device = {param.device}"
            )

            print(
                f"  dtype  = {param.dtype}"
            )

            print(
                f"  shape  = {tuple(param.shape)}"
            )

    if not found:

        raise RuntimeError(
            "没有发现 LoRA 参数！"
        )


# ============================================================
# 7. 参数统计
# ============================================================

def count_parameters(
    model
):

    total_params = 0

    trainable_params = 0

    for param in model.parameters():

        total_params += (
            param.numel()
        )

        if param.requires_grad:

            trainable_params += (
                param.numel()
            )

    return (
        total_params,
        trainable_params
    )


# ============================================================
# 8. 加载 CLIP
# ============================================================

def load_clip_model():

    if not os.path.exists(
        MODEL_PATH
    ):

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

    print(
        f"Original CLIP dtype: "
        f"{model.dtype}"
    )

    print("=" * 70)

    return (
        model,
        preprocess
    )


# ============================================================
# 9. ImageNet 类别映射
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

        synset_to_name[
            wnid
        ] = class_name

    return (
        synset_to_name
    )


# ============================================================
# 10. 数据集
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
    # 类别一致性
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
            f"理论上应该是 "
            f"{expected_train_size}"
        )

    print(
        "\n数据集信息："
    )

    print(
        f"Train images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Val images  : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test images : "
        f"{len(test_dataset)}"
    )

    print(
        f"Classes     : "
        f"{len(train_dataset.classes)}"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# 11. DataLoader
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
# 12. 构造 Text Features
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
        "\nPrompt 示例："
    )

    for prompt in prompts[:10]:

        print(
            f"  {prompt}"
        )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    text_tokens = (
        clip.tokenize(
            prompts
        ).to(DEVICE)
    )

    # --------------------------------------------------------
    # CLIP Text Encoder
    # 完全冻结
    # --------------------------------------------------------

    text_features = (
        model.encode_text(
            text_tokens
        )
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
# 13. 单 epoch 训练
# ============================================================

def train_one_epoch(
    model,
    loader,
    text_features,
    logit_scale,
    optimizer,
    criterion,
    scaler,
    amp_enabled
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(
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

        # ----------------------------------------------------
        # CLIP Vision Encoder + LoRA
        # ----------------------------------------------------

        with torch.autocast(
            device_type=DEVICE,
            enabled=amp_enabled
        ):

            image_features = (
                model.encode_image(
                    images
                )
            )

            # ------------------------------------------------
            # 转 FP32
            # ------------------------------------------------

            image_features = (
                image_features.float()
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Image-Text Similarity
            # ------------------------------------------------

            logits = (
                logit_scale
                * (
                    image_features
                    @ text_features.T
                )
            )

            loss = (
                criterion(
                    logits,
                    labels
                )
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += (
            labels.size(0)
        )

    avg_loss = (
        total_loss
        / total
    )

    accuracy = (
        100.0
        * correct
        / total
    )

    return (
        avg_loss,
        accuracy
    )


# ============================================================
# 14. Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    text_features,
    logit_scale,
    description
):

    model.eval()

    correct = 0

    total = 0

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

    for images, labels in tqdm(
        loader,
        desc=f"{description} Evaluation"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE,
            enabled=(DEVICE == "cuda")
        ):

            image_features = (
                model.encode_image(
                    images
                )
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

        logits = (
            logit_scale
            * (
                image_features
                @ text_features.T
            )
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += (
            labels.size(0)
        )

        # ----------------------------------------------------
        # Per-class Accuracy
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Overall Accuracy
    # --------------------------------------------------------

    accuracy = (
        100.0
        * correct
        / total
    )

    # --------------------------------------------------------
    # Mean Class Accuracy
    # --------------------------------------------------------

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
# 15. 保存 LoRA Checkpoint
# ============================================================

def save_lora_checkpoint(
    model,
    epoch
):

    os.makedirs(
        "models",
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        "models",
        "clip_vit_b32_lora.pth"
    )

    lora_state = {}

    for name, param in (
        model.named_parameters()
    ):

        if (
            "lora_A" in name
            or
            "lora_B" in name
        ):

            lora_state[name] = (
                param.detach()
                .cpu()
                .clone()
            )

    torch.save(
        {
            "epoch": epoch,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "state_dict": lora_state
        },
        checkpoint_path
    )

    print(
        "\nLoRA checkpoint 已保存："
        f"\n{checkpoint_path}"
    )


# ============================================================
# 16. 保存结果
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
            "CLIP LoRA Baseline\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        f.write(
            "Experiment: clip_lora\n"
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
            "Original CLIP Weights Frozen: Yes\n"
        )

        f.write(
            "Text Encoder Frozen: Yes\n"
        )

        # ----------------------------------------------------
        # LoRA
        # ----------------------------------------------------

        f.write(
            "Method: LoRA\n"
        )

        f.write(
            "Adaptation Side: Vision Encoder\n"
        )

        f.write(
            "Target Layers: Visual Transformer MLP "
            "(c_fc + c_proj)\n"
        )

        f.write(
            f"LoRA Rank: {LORA_RANK}\n"
        )

        f.write(
            f"LoRA Alpha: {LORA_ALPHA}\n"
        )

        f.write(
            f"LoRA Dropout: {LORA_DROPOUT}\n"
        )

        f.write(
            f"Total Parameters: {total_params}\n"
        )

        f.write(
            f"Trainable LoRA Parameters: "
            f"{trainable_params}\n"
        )

        f.write(
            f"Trainable Ratio: "
            f"{100.0 * trainable_params / total_params:.4f}%\n"
        )

        # ----------------------------------------------------
        # Dataset
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
        "\n实验结果已保存到："
        f"{results_file}"
    )


# ============================================================
# 17. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("CLIP LoRA Few-shot Adaptation")
    print("=" * 70)

    print(
        "Pretrained CLIP ViT-B/32"
    )

    print(
        "        ↓"
    )

    print(
        "Frozen Original Parameters"
    )

    print(
        "        ↓"
    )

    print(
        "LoRA on Visual Transformer MLP"
    )

    print(
        "        ↓"
    )

    print(
        f"{K_SHOT}-shot Adaptation"
    )

    print(
        "        ↓"
    )

    print(
        "Image-Text Classification"
    )

    print("=" * 70)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"K-shot       : {K_SHOT}"
    )

    print(
        f"LoRA Rank    : {LORA_RANK}"
    )

    print(
        f"LoRA Alpha   : {LORA_ALPHA}"
    )

    print(
        f"Learning Rate: {LR}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )


    # ========================================================
    # 1. 加载 CLIP
    # ========================================================

    model, preprocess = (
        load_clip_model()
    )


    # ========================================================
    # 2. 注入 LoRA
    # ========================================================

    replaced_layers = (
        inject_lora_into_visual_mlp(
            model=model,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT
        )
    )

    print(
        f"\nLoRA 注入数量: "
        f"{len(replaced_layers)}"
    )


    # ========================================================
    # 3. 冻结原始 CLIP
    # ========================================================

    trainable_names = (
        freeze_clip_except_lora(
            model
        )
    )

    print(
        f"\n可训练参数数量: "
        f"{len(trainable_names)}"
    )


    # ========================================================
    # 4. 确保模型 device 正确
    #
    # LoRA 参数已经在正确 device，
    # 这里再统一检查一次。
    # ========================================================

    model = model.to(
        DEVICE
    )


    # ========================================================
    # 5. 检查 LoRA 参数
    # ========================================================

    inspect_lora_parameters(
        model
    )


    # ========================================================
    # 6. 参数统计
    # ========================================================

    (
        total_params,
        trainable_params
    ) = count_parameters(
        model
    )

    print(
        "\n模型参数统计："
    )

    print(
        f"Total Parameters: "
        f"{total_params / 1e6:.2f} M"
    )

    print(
        f"Trainable Parameters: "
        f"{trainable_params / 1e3:.2f} K"
    )

    print(
        f"Trainable Ratio: "
        f"{100.0 * trainable_params / total_params:.4f}%"
    )


    # ========================================================
    # 7. ImageNet 类别映射
    # ========================================================

    synset_to_name = (
        load_imagenet_mapping()
    )


    # ========================================================
    # 8. 数据集
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets(
        preprocess
    )


    # ========================================================
    # 9. 类别名称
    # ========================================================

    class_names = []

    for synset in train_dataset.classes:

        if synset not in synset_to_name:

            raise KeyError(
                f"{synset} 不存在于 "
                f"imagenet_class_index.json"
            )

        class_names.append(
            synset_to_name[
                synset
            ]
        )

    print(
        "\n类别名称示例："
    )

    for name in class_names[:10]:

        print(
            f"  {name}"
        )


    # ========================================================
    # 10. Text Features
    #
    # Text Encoder 完全冻结
    # ========================================================

    text_features = (
        build_text_features(
            model=model,
            class_names=class_names
        )
    )


    # ========================================================
    # 11. DataLoader
    # ========================================================

    train_loader = build_dataloader(
        train_dataset,
        shuffle=True
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
    # 12. Optimizer
    #
    # 只优化 LoRA A/B
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(trainable_parameters) == 0:

        raise RuntimeError(
            "没有任何可训练参数！"
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = (
        nn.CrossEntropyLoss()
    )


    # ========================================================
    # 13. Scheduler
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )


    # ========================================================
    # 14. AMP
    # ========================================================

    amp_enabled = (
        DEVICE == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )


    # ========================================================
    # 15. Logit Scale
    #
    # CLIP 原始 logit_scale 冻结
    # ========================================================

    logit_scale = (
        model.logit_scale
        .exp()
        .float()
        .detach()
    )


    # ========================================================
    # 16. 最佳模型
    # ========================================================

    best_val_acc = 0.0

    best_epoch = 0

    best_lora_state = {}

    print(
        "\n开始 LoRA Few-shot Training...\n"
    )


    # ========================================================
    # 17. Training Loop
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        (
            train_loss,
            train_acc
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            text_features=text_features,
            logit_scale=logit_scale,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            amp_enabled=amp_enabled
        )

        scheduler.step()


        # ====================================================
        # Validation
        # ====================================================

        if (
            (epoch + 1) % EVAL_FREQ == 0
            or
            (epoch + 1) == EPOCHS
        ):

            (
                val_acc,
                val_mean_class_acc
            ) = evaluate(
                model=model,
                loader=val_loader,
                text_features=text_features,
                logit_scale=logit_scale,
                description="Validation"
            )

            # ------------------------------------------------
            # 保存最佳 LoRA
            # ------------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = (
                    val_acc
                )

                best_epoch = (
                    epoch + 1
                )

                best_lora_state = {}

                for name, param in (
                    model.named_parameters()
                ):

                    if (
                        "lora_A" in name
                        or
                        "lora_B" in name
                    ):

                        best_lora_state[
                            name
                        ] = (
                            param.detach()
                            .cpu()
                            .clone()
                        )

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Train Loss: "
                f"{train_loss:.4f} "
                f"| Train Acc: "
                f"{train_acc:.2f}% "
                f"| Val Acc: "
                f"{val_acc:.2f}% "
                f"| LR: "
                f"{scheduler.get_last_lr()[0]:.6f}"
            )

        else:

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Train Loss: "
                f"{train_loss:.4f} "
                f"| Train Acc: "
                f"{train_acc:.2f}% "
                f"| LR: "
                f"{scheduler.get_last_lr()[0]:.6f}"
            )


    # ========================================================
    # 18. 恢复最佳 LoRA
    # ========================================================

    if len(best_lora_state) == 0:

        raise RuntimeError(
            "没有保存到最佳 LoRA 参数。"
        )

    print(
        "\n正在恢复最佳 LoRA 参数..."
    )

    with torch.no_grad():

        for name, param in (
            model.named_parameters()
        ):

            if name in best_lora_state:

                param.copy_(
                    best_lora_state[
                        name
                    ].to(
                        device=param.device,
                        dtype=param.dtype
                    )
                )


    # ========================================================
    # 19. 保存 LoRA checkpoint
    # ========================================================

    save_lora_checkpoint(
        model=model,
        epoch=best_epoch
    )


    # ========================================================
    # 20. Final Validation
    # ========================================================

    (
        final_val_acc,
        final_val_mean_class_acc
    ) = evaluate(
        model=model,
        loader=val_loader,
        text_features=text_features,
        logit_scale=logit_scale,
        description="Final Validation"
    )


    # ========================================================
    # 21. Test
    # ========================================================

    (
        test_acc,
        test_mean_class_acc
    ) = evaluate(
        model=model,
        loader=test_loader,
        text_features=text_features,
        logit_scale=logit_scale,
        description="Test"
    )


    # ========================================================
    # 22. 保存结果
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
        test_size=len(test_dataset)
    )


    # ========================================================
    # 23. 最终输出
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "CLIP LoRA 实验结束"
    )
    print("=" * 70)

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
        f"Trainable Parameters: "
        f"{trainable_params / 1e3:.2f} K"
    )

    print(
        f"Trainable Ratio: "
        f"{100.0 * trainable_params / total_params:.4f}%"
    )

    print("=" * 70)


# ============================================================
# 24. 程序入口
# ============================================================

if __name__ == "__main__":

    main()