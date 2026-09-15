import os
import torch
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# 配置
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_pretrained.pth"
)


# ============================================================
# 下载并保存预训练 ResNet18
# ============================================================

def download_pretrained_resnet18():

    # 创建 models 文件夹
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 如果模型已经存在，则不重复下载
    if os.path.exists(MODEL_PATH):
        print("=" * 70)
        print("预训练 ResNet18 已存在，无需重复下载。")
        print(f"模型路径: {MODEL_PATH}")
        print("=" * 70)
        return

    print("=" * 70)
    print("开始下载 ImageNet 预训练 ResNet18...")
    print("=" * 70)

    # 使用 torchvision 官方提供的 ImageNet 预训练权重
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    # 只保存模型参数，而不是整个模型对象
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\n预训练模型下载完成！")
    print(f"模型路径: {MODEL_PATH}")

    # 输出权重来源
    print(
        f"Weight source: {weights}"
    )

    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    download_pretrained_resnet18()