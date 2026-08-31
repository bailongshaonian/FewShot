import os
import json
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
# CoOp 参数
# ============================================================

N_CTX = 4

# 使用自然语言初始化 Context
CTX_INIT = "a photo of a"


# ============================================================
# Training 参数
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100

LR = 0.002
WEIGHT_DECAY = 1e-4

EVAL_FREQ = 5


# ============================================================
# DataLoader
# ============================================================

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
# 3. ImageNet 类别映射
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
# 4. 加载 OpenAI CLIP
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
    # 冻结全部 CLIP 参数
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
    # 检查 20-shot
    # --------------------------------------------------------

    expected_train_size = (
        NUM_CLASSES * K_SHOT
    )

    if len(train_dataset) != expected_train_size:

        raise ValueError(
            f"Train Size={len(train_dataset)}，"
            f"理论上应该是 "
            f"{NUM_CLASSES} × {K_SHOT} = "
            f"{expected_train_size}"
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
# 6. 提取冻结 CLIP Image Features
# ============================================================

@torch.no_grad()
def extract_image_features(
    model,
    dataset,
    description
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=False
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

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        image_features = (
            model.encode_image(images)
        )

        # ----------------------------------------------------
        # OpenAI CLIP 可能输出 FP16
        # 后续统一使用 FP32
        # ----------------------------------------------------

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

        all_features.append(
            image_features.cpu()
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
# 7. CoOp Prompt Learner
# ============================================================

class CoOpPromptLearner(nn.Module):

    """
    CoOp-style Prompt Learner

    Prompt：

        [SOS]
        [CTX1] [CTX2] ... [CTX_N]
        [CLASS]
        [EOS]
        [PAD ...]

    CLIP 本身完全冻结。
    只有 self.ctx 是可训练参数。
    """

    def __init__(
        self,
        model,
        class_names,
        n_ctx=4,
        ctx_init=None
    ):

        super().__init__()

        self.model = model
        self.class_names = class_names
        self.n_ctx = n_ctx

        # ----------------------------------------------------
        # CLIP 参数
        # ----------------------------------------------------

        self.token_embedding = (
            model.token_embedding
        )

        self.positional_embedding = (
            model.positional_embedding
        )

        self.transformer = (
            model.transformer
        )

        self.ln_final = (
            model.ln_final
        )

        self.text_projection = (
            model.text_projection
        )

        self.context_length = (
            model.positional_embedding.shape[0]
        )

        self.clip_dtype = model.dtype

        # ----------------------------------------------------
        # Context dimension
        # ----------------------------------------------------

        ctx_dim = (
            model.token_embedding.weight.shape[1]
        )

        self.ctx_dim = ctx_dim

        # ----------------------------------------------------
        # 1. 初始化 Context
        # ----------------------------------------------------

        if ctx_init is not None:

            init_token_ids = clip.tokenize(
                ctx_init
            ).to(DEVICE)

            with torch.no_grad():

                init_embeddings = (
                    self.token_embedding(
                        init_token_ids
                    )
                    .squeeze(0)
                )

            # token 0 = SOS
            #
            # 后面开始是 ctx_init 的 token
            # 找 EOS 之前的部分

            eos_id = 49407

            token_ids = (
                init_token_ids
                .squeeze(0)
                .tolist()
            )

            if eos_id in token_ids:

                eos_position = (
                    token_ids.index(
                        eos_id
                    )
                )

            else:

                eos_position = (
                    len(token_ids)
                )

            init_context = (
                init_embeddings[
                    1:eos_position
                ]
            )

            if (
                init_context.shape[0]
                >= n_ctx
            ):

                ctx = (
                    init_context[
                        :n_ctx
                    ]
                    .float()
                    .clone()
                )

            else:

                ctx = torch.empty(
                    n_ctx,
                    ctx_dim,
                    dtype=torch.float32
                )

                # 先复制已有初始化
                if init_context.shape[0] > 0:

                    ctx[
                        :init_context.shape[0]
                    ] = (
                        init_context.float()
                    )

                # 剩余部分随机初始化
                remaining = (
                    n_ctx
                    - init_context.shape[0]
                )

                if remaining > 0:

                    nn.init.normal_(
                        ctx[
                            init_context.shape[0]:
                        ],
                        std=0.02
                    )

        else:

            ctx = torch.empty(
                n_ctx,
                ctx_dim,
                dtype=torch.float32
            )

            nn.init.normal_(
                ctx,
                std=0.02
            )

        # ----------------------------------------------------
        # 关键：
        #
        # CoOp Context 始终保持 FP32
        # ----------------------------------------------------

        self.ctx = nn.Parameter(
            ctx.float()
        )

        # ----------------------------------------------------
        # 2. 为每一个类别准备固定 token embedding
        # ----------------------------------------------------

        prefix_list = []
        suffix_list = []
        eos_positions = []

        with torch.no_grad():

            for class_name in class_names:

                token_ids = clip.tokenize(
                    class_name
                ).to(DEVICE)

                token_ids = (
                    token_ids.squeeze(0)
                )

                # --------------------------------------------
                # 找 EOS
                # --------------------------------------------

                token_list = (
                    token_ids.tolist()
                )

                if 49407 in token_list:

                    eos_position = (
                        token_list.index(
                            49407
                        )
                    )

                else:

                    raise RuntimeError(
                        f"类别 {class_name} "
                        f"没有找到 EOS token！"
                    )

                # --------------------------------------------
                # prefix:
                #
                # [SOS]
                # --------------------------------------------

                prefix_ids = (
                    token_ids[
                        :1
                    ]
                    .unsqueeze(0)
                )

                prefix_embedding = (
                    self.token_embedding(
                        prefix_ids
                    )
                    .squeeze(0)
                )

                # --------------------------------------------
                # suffix:
                #
                # [CLASS TOKENS][EOS]
                # --------------------------------------------

                suffix_ids = (
                    token_ids[
                        1:eos_position + 1
                    ]
                    .unsqueeze(0)
                )

                suffix_embedding = (
                    self.token_embedding(
                        suffix_ids
                    )
                    .squeeze(0)
                )

                prefix_list.append(
                    prefix_embedding
                )

                suffix_list.append(
                    suffix_embedding
                )

                # --------------------------------------------
                # 最终 EOS 位置：
                #
                # SOS
                # + N_CTX
                # + CLASS TOKENS
                #
                # --------------------------------------------

                num_class_tokens = (
                    suffix_embedding.shape[0]
                    - 1
                )

                final_eos_position = (
                    1
                    + n_ctx
                    + num_class_tokens
                )

                eos_positions.append(
                    final_eos_position
                )

        # ----------------------------------------------------
        # 注册为 buffer
        #
        # 它们不会被 optimizer 更新
        # ----------------------------------------------------

        self.register_buffer(
            "prefix",
            torch.stack(
                prefix_list
            )
        )

        self.register_buffer(
            "suffix",
            self._pad_suffixes(
                suffix_list
            )
        )

        self.eos_positions = (
            eos_positions
        )

    # --------------------------------------------------------
    # Padding suffix
    # --------------------------------------------------------

    def _pad_suffixes(
        self,
        suffix_list
    ):

        max_length = max(
            x.shape[0]
            for x in suffix_list
        )

        max_length = min(
            max_length,
            self.context_length
        )

        batch_size = len(
            suffix_list
        )

        dtype = suffix_list[0].dtype

        suffix_tensor = torch.zeros(
            batch_size,
            max_length,
            self.ctx_dim,
            dtype=dtype,
            device=suffix_list[0].device
        )

        for i, suffix in enumerate(
            suffix_list
        ):

            length = min(
                suffix.shape[0],
                max_length
            )

            suffix_tensor[
                i,
                :length
            ] = suffix[
                :length
            ]

        return suffix_tensor

    # --------------------------------------------------------
    # Build Prompt
    # --------------------------------------------------------

    def build_prompts(self):

        # ----------------------------------------------------
        # 关键：
        # ctx 是 FP32
        # prefix / suffix 是 CLIP dtype
        # 我们先把 ctx 转成 CLIP dtype
        # ----------------------------------------------------

        ctx = self.ctx.to(
            dtype=self.clip_dtype
        )

        batch_size = (
            self.prefix.shape[0]
        )

        prompt_list = []

        for i in range(
            batch_size
        ):

            prefix = (
                self.prefix[
                    i:i + 1
                ]
            )

            suffix = (
                self.suffix[
                    i:i + 1
                ]
            )

            # ------------------------------------------------
            # [SOS] + [CTX] + [CLASS][EOS]
            # ------------------------------------------------

            prompt = torch.cat(
                [
                    prefix,
                    ctx.unsqueeze(0),
                    suffix
                ],
                dim=1
            )

            # ------------------------------------------------
            # CLIP context length = 77
            # ------------------------------------------------

            if (
                prompt.shape[1]
                > self.context_length
            ):

                prompt = prompt[
                    :,
                    :self.context_length,
                    :
                ]

            elif (
                prompt.shape[1]
                < self.context_length
            ):

                padding = torch.zeros(
                    1,
                    self.context_length
                    - prompt.shape[1],
                    self.ctx_dim,
                    dtype=prompt.dtype,
                    device=prompt.device
                )

                prompt = torch.cat(
                    [
                        prompt,
                        padding
                    ],
                    dim=1
                )

            prompt_list.append(
                prompt
            )

        return torch.cat(
            prompt_list,
            dim=0
        )

    # --------------------------------------------------------
    # Encode Text
    # --------------------------------------------------------

    def encode_text(self):

        prompts = self.build_prompts()

        # ----------------------------------------------------
        # 统一 dtype
        # ----------------------------------------------------

        prompts = prompts.to(
            dtype=self.clip_dtype
        )

        positional_embedding = (
            self.positional_embedding
            .to(
                dtype=self.clip_dtype,
                device=prompts.device
            )
        )

        # ----------------------------------------------------
        # Positional Embedding
        # ----------------------------------------------------

        x = (
            prompts
            + positional_embedding
        )

        # ----------------------------------------------------
        # Transformer
        #
        # [N, L, D]
        # →
        # [L, N, D]
        # ----------------------------------------------------

        x = x.permute(
            1,
            0,
            2
        )

        x = self.transformer(
            x
        )

        x = x.permute(
            1,
            0,
            2
        )

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.ln_final(
            x
        )

        # ----------------------------------------------------
        # Text Feature
        #
        # 每个类别使用自己的 EOS position
        # ----------------------------------------------------

        text_features = []

        for i, eos_position in enumerate(
            self.eos_positions
        ):

            feature = (
                x[
                    i,
                    eos_position
                ]
            )

            text_features.append(
                feature
            )

        text_features = torch.stack(
            text_features
        )

        # ----------------------------------------------------
        # Projection
        #
        # 此时两者都为 CLIP dtype
        # ----------------------------------------------------

        text_projection = (
            self.text_projection
        )

        text_features = (
            text_features
            @ text_projection
        )

        # ----------------------------------------------------
        # 最终转 FP32
        # ----------------------------------------------------

        text_features = (
            text_features.float()
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

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

        return text_features


# ============================================================
# 8. 训练 CoOp
# ============================================================

def train_coop(
    model,
    prompt_learner,
    train_features,
    train_labels,
    val_features,
    val_labels
):

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    # --------------------------------------------------------
    # 只优化 ctx
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    prompt_learner.ctx
                ],
                "lr": LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # Best Model
    # --------------------------------------------------------

    best_val_acc = 0.0
    best_epoch = 0

    best_ctx = (
        prompt_learner.ctx
        .detach()
        .clone()
    )

    # --------------------------------------------------------
    # CLIP logit scale
    # --------------------------------------------------------

    logit_scale = (
        model.logit_scale.exp()
        .float()
        .detach()
    )

    print(
        "\n开始 CoOp Prompt Training...\n"
    )

    for epoch in range(
        EPOCHS
    ):

        prompt_learner.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # ----------------------------------------------------
        # 每个 epoch 先计算一次 text features
        # ----------------------------------------------------

        text_features = (
            prompt_learner.encode_text()
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

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
            # Image-Text Similarity
            # ------------------------------------------------

            logits = (
                logit_scale
                * (
                    features
                    @ text_features.T
                )
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

            # ------------------------------------------------
            # Prompt 更新后重新计算 Text Feature
            # ------------------------------------------------
            #
            # 这是为了保证后续 batch 使用最新的 ctx。
            #
            # ------------------------------------------------

            text_features = (
                prompt_learner.encode_text()
            )

        scheduler.step()

        train_loss = (
            total_loss
            / total
        )

        train_acc = (
            100.0
            * correct
            / total
        )

        # ====================================================
        # Validation
        # ====================================================

        if (
            (epoch + 1) % EVAL_FREQ == 0
            or
            (epoch + 1) == EPOCHS
        ):

            prompt_learner.eval()

            with torch.no_grad():

                val_text_features = (
                    prompt_learner.encode_text()
                )

                val_features_device = (
                    val_features.to(
                        DEVICE
                    )
                )

                val_labels_device = (
                    val_labels.to(
                        DEVICE
                    )
                )

                val_logits = (
                    logit_scale
                    * (
                        val_features_device
                        @ val_text_features.T
                    )
                )

                val_predictions = (
                    val_logits.argmax(
                        dim=1
                    )
                )

                val_acc = (
                    100.0
                    * (
                        val_predictions
                        == val_labels_device
                    )
                    .sum()
                    .item()
                    /
                    len(val_labels_device)
                )

            # ------------------------------------------------
            # 保存最佳 Context
            # ------------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                best_epoch = (
                    epoch + 1
                )

                best_ctx = (
                    prompt_learner.ctx
                    .detach()
                    .clone()
                )

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.2f}% "
                f"| Val Acc: {val_acc:.2f}% "
                f"| LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        else:

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
                f"| Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.2f}% "
                f"| LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    # --------------------------------------------------------
    # 恢复最佳 Context
    # --------------------------------------------------------

    with torch.no_grad():

        prompt_learner.ctx.copy_(
            best_ctx
        )

    return (
        prompt_learner,
        best_val_acc,
        best_epoch
    )


# ============================================================
# 9. Evaluation
# ============================================================

@torch.no_grad()
def evaluate_coop(
    model,
    prompt_learner,
    features,
    labels,
    description
):

    prompt_learner.eval()

    text_features = (
        prompt_learner.encode_text()
    )

    logit_scale = (
        model.logit_scale.exp()
        .float()
        .detach()
    )

    features = features.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    logits = (
        logit_scale
        * (
            features
            @ text_features.T
        )
    )

    predictions = (
        logits.argmax(
            dim=1
        )
    )

    accuracy = (
        100.0
        * (
            predictions == labels
        )
        .sum()
        .item()
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
# 10. 保存结果
# ============================================================

def save_results(
    best_val_acc,
    best_epoch,
    final_val_acc,
    val_mean_class_acc,
    test_acc,
    test_mean_class_acc,
    train_size,
    val_size,
    test_size,
    trainable_params
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
            "CLIP CoOp Baseline\n"
        )
        f.write("=" * 70 + "\n")

        # ----------------------------------------------------
        # 实验
        # ----------------------------------------------------

        f.write(
            "Experiment: clip_coop\n"
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
            "CLIP Image Encoder Frozen: Yes\n"
        )

        f.write(
            "CLIP Text Encoder Frozen: Yes\n"
        )

        # ----------------------------------------------------
        # CoOp
        # ----------------------------------------------------

        f.write(
            "Method: CoOp-style Prompt Tuning\n"
        )

        f.write(
            f"Context Tokens: {N_CTX}\n"
        )

        f.write(
            f"Context Initialization: "
            f'"{CTX_INIT}"\n'
        )

        f.write(
            "Prompt Type: "
            "Continuous Learnable Context\n"
        )

        f.write(
            "Learnable Parameters: Context Only\n"
        )

        f.write(
            f"Trainable Parameters: "
            f"{trainable_params}\n"
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
        f"{results_file}"
    )


# ============================================================
# 11. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("CLIP CoOp Few-shot Adaptation")
    print("=" * 70)

    print(
        "CLIP ViT-B/32"
    )

    print(
        "      ↓"
    )

    print(
        "Frozen Image / Text Encoder"
    )

    print(
        "      ↓"
    )

    print(
        "Learnable Context Tokens"
    )

    print(
        "      ↓"
    )

    print(
        f"{K_SHOT}-shot Training"
    )

    print(
        "      ↓"
    )

    print(
        "Few-shot Classification"
    )

    print("=" * 70)

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"K-shot        : {K_SHOT}"
    )

    print(
        f"Context Tokens: {N_CTX}"
    )

    print(
        f"Context Init  : {CTX_INIT}"
    )

    print(
        f"Epochs        : {EPOCHS}"
    )

    print(
        f"Learning Rate : {LR}"
    )


    # ========================================================
    # 1. CLIP
    # ========================================================

    model, preprocess = (
        load_clip_model()
    )


    # ========================================================
    # 2. 类别映射
    # ========================================================

    synset_to_name = (
        load_imagenet_mapping()
    )


    # ========================================================
    # 3. 数据集
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets(
        preprocess
    )


    # ========================================================
    # 4. 构造可读类别名
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
    # 5. 提取 Image Features
    #
    # CLIP 完全冻结，因此只提取一次。
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
    # 6. Prompt Learner
    # ========================================================

    prompt_learner = (
        CoOpPromptLearner(
            model=model,
            class_names=class_names,
            n_ctx=N_CTX,
            ctx_init=CTX_INIT
        )
    ).to(DEVICE)


    # ========================================================
    # 7. 参数统计
    # ========================================================

    trainable_params = sum(
        p.numel()
        for p in prompt_learner.parameters()
        if p.requires_grad
    )

    print(
        "\nCoOp 参数信息："
    )

    print(
        f"Context Dimension: "
        f"{prompt_learner.ctx_dim}"
    )

    print(
        f"Context Tokens: "
        f"{N_CTX}"
    )

    print(
        f"Trainable Params: "
        f"{trainable_params}"
    )


    # ========================================================
    # 8. CoOp Training
    # ========================================================

    (
        prompt_learner,
        best_val_acc,
        best_epoch
    ) = train_coop(
        model=model,
        prompt_learner=prompt_learner,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels
    )


    # ========================================================
    # 9. Final Validation
    # ========================================================

    (
        final_val_acc,
        val_mean_class_acc
    ) = evaluate_coop(
        model=model,
        prompt_learner=prompt_learner,
        features=val_features,
        labels=val_labels,
        description="Validation"
    )


    # ========================================================
    # 10. Test
    # ========================================================

    (
        test_acc,
        test_mean_class_acc
    ) = evaluate_coop(
        model=model,
        prompt_learner=prompt_learner,
        features=test_features,
        labels=test_labels,
        description="Test"
    )


    # ========================================================
    # 11. 保存
    # ========================================================

    save_results(
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        final_val_acc=final_val_acc,
        val_mean_class_acc=val_mean_class_acc,
        test_acc=test_acc,
        test_mean_class_acc=test_mean_class_acc,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        trainable_params=trainable_params
    )


    # ========================================================
    # 12. 最终输出
    # ========================================================

    print("\n" + "=" * 70)
    print("CLIP CoOp 实验结束")
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

    print("=" * 70)


# ============================================================
# 13. 程序入口
# ============================================================

if __name__ == "__main__":
    main()