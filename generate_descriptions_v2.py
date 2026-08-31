import json
import os

import nltk
from nltk.corpus import wordnet as wn


# ============================================================
# 1. 配置
# ============================================================

CLASS_INDEX_PATH = os.path.join(
    "mini-imagenet",
    "imagenet_class_index.json"
)

OUTPUT_PATH = "class_descriptions.json"


# ============================================================
# 2. 准备 WordNet
# ============================================================

def prepare_wordnet():

    print("=" * 70)
    print("检查 WordNet 数据...")
    print("=" * 70)

    try:
        # 测试 WordNet 是否可以正常访问
        wn.synsets("dog")

        print("WordNet 数据已经存在。")

    except LookupError:

        print("未找到 WordNet 数据，开始下载...")

        nltk.download(
            "wordnet",
            quiet=False
        )

        nltk.download(
            "omw-1.4",
            quiet=False
        )

    print()


# ============================================================
# 3. ImageNet WNID → WordNet Synset
# ============================================================

def imagenet_wnid_to_synset(wnid):
    """
    ImageNet WNID 示例：

        n01440764

    其中：
        n  -> noun
        01440764 -> WordNet offset

    使用 NLTK：
        wn.synset_from_pos_and_offset("n", 1440764)

    获取：
        Synset('tench.n.01')
    """

    if not isinstance(wnid, str):
        return None

    # ImageNet 这里主要是 noun
    if not wnid.startswith("n"):
        return None

    try:

        offset = int(
            wnid[1:]
        )

    except ValueError:

        return None

    try:

        synset = wn.synset_from_pos_and_offset(
            "n",
            offset
        )

        return synset

    except Exception:

        return None


# ============================================================
# 4. 生成类别描述
# ============================================================

def generate_descriptions():

    if not os.path.exists(
        CLASS_INDEX_PATH
    ):

        raise FileNotFoundError(
            f"找不到文件：\n"
            f"{CLASS_INDEX_PATH}"
        )

    # --------------------------------------------------------
    # 读取 ImageNet class index
    # --------------------------------------------------------

    with open(
        CLASS_INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_index = json.load(f)

    print(
        f"读取到 {len(class_index)} 个 ImageNet 类别。"
    )

    descriptions = {}

    success_count = 0
    failed_count = 0

    failed_items = []

    # ========================================================
    # 遍历类别
    # ========================================================

    for _, value in class_index.items():

        if (
            not isinstance(value, list)
            or len(value) < 2
        ):
            continue

        # ----------------------------------------------------
        # ImageNet 信息
        # ----------------------------------------------------

        wnid = value[0]
        class_name = value[1]

        clean_name = (
            class_name
            .replace("_", " ")
            .replace("-", " ")
        )

        # ----------------------------------------------------
        # 精确获取 WordNet Synset
        # ----------------------------------------------------

        synset = (
            imagenet_wnid_to_synset(
                wnid
            )
        )

        if synset is None:

            failed_count += 1

            failed_items.append(
                {
                    "wnid": wnid,
                    "class_name": clean_name
                }
            )

            # 保留一个明确的 fallback
            descriptions[clean_name] = {
                "wnid": wnid,
                "synset": None,
                "definition": None,
                "description": (
                    "which is a type of "
                    "object or living thing."
                )
            }

            continue

        # ----------------------------------------------------
        # WordNet definition
        # ----------------------------------------------------

        definition = (
            synset.definition()
        )

        descriptions[clean_name] = {
            "wnid": wnid,
            "synset": synset.name(),
            "definition": definition,
            "description": (
                f"which is {definition}."
            )
        }

        success_count += 1

    return (
        descriptions,
        success_count,
        failed_count,
        failed_items
    )


# ============================================================
# 5. 保存 JSON
# ============================================================

def save_descriptions(
    descriptions
):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            descriptions,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# 6. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("ImageNet → WordNet 类别描述生成器")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. WordNet
    # --------------------------------------------------------

    prepare_wordnet()

    # --------------------------------------------------------
    # 2. Generate
    # --------------------------------------------------------

    (
        descriptions,
        success_count,
        failed_count,
        failed_items
    ) = generate_descriptions()

    # --------------------------------------------------------
    # 3. Save
    # --------------------------------------------------------

    save_descriptions(
        descriptions
    )

    # ========================================================
    # 4. 输出统计
    # ========================================================

    print("\n" + "=" * 70)
    print("类别描述生成完成")
    print("=" * 70)

    print(
        f"总类别数      : "
        f"{len(descriptions)}"
    )

    print(
        f"WordNet 成功数 : "
        f"{success_count}"
    )

    print(
        f"WordNet 失败数 : "
        f"{failed_count}"
    )

    print(
        f"输出文件       : "
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # 5. 显示失败项
    # ========================================================

    if failed_items:

        print("\n>>> 未找到 WordNet 的类别：")

        for item in failed_items:

            print(
                f"  {item['wnid']} "
                f"-> "
                f"{item['class_name']}"
            )

    # ========================================================
    # 6. Preview
    # ========================================================

    print("\n>>> 类别描述预览：")

    preview_count = 0

    for class_name, info in descriptions.items():

        if preview_count >= 10:
            break

        print("\n" + "-" * 60)

        print(
            f"Class Name: "
            f"{class_name}"
        )

        print(
            f"WNID: "
            f"{info['wnid']}"
        )

        print(
            f"Synset: "
            f"{info['synset']}"
        )

        print(
            f"Definition: "
            f"{info['definition']}"
        )

        print(
            "Hard Prompt: "
            f"a photo of a {class_name}, "
            f"{info['description']}"
        )

        preview_count += 1

    print("\n" + "=" * 70)


# ============================================================
# 7. 程序入口
# ============================================================

if __name__ == "__main__":
    main()