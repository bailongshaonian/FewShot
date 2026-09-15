import json
import os
import nltk
from nltk.corpus import wordnet as wn

def main():
    print("正在下載 WordNet 詞典數據 (只需下載一次)...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    # 確保路徑與你的項目結構一致
    json_path = "./mini-imagenet/imagenet_class_index.json"
    out_path = os.path.join("mini-imagenet", "class_descriptions.json")

    with open(json_path, 'r', encoding='utf-8') as f:
        class_index = json.load(f)

    descriptions = {}

    for key, value in class_index.items():
        wnid = value[0]    # 例如 'n02099601'
        name = value[1]    # 例如 'golden_retriever'
        clean_name = name.replace('_', ' ')

        # 核心黑科技：去 WordNet 詞典裡查詢這個詞的意思
        synsets = wn.synsets(name)

        if synsets:
            # 獲取這個詞最常用（第一個）的官方定義
            definition = synsets[0].definition()
            # 構造成一句順口的話，比如 "which is an English breed..."
            descriptions[clean_name] = f"which is {definition}."
        else:
            # 極少數生僻詞的兜底策略
            descriptions[clean_name] = "which is a kind of object or living thing."

    # 將結果保存為 JSON
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=4, ensure_ascii=False)

    print(f"\n成功為 {len(descriptions)} 個類別自動生成了詳細描述！")
    print(f"文件已保存至: {out_path}\n")

    # 預覽幾個生成出來的句子
    print(">>> 完美硬提示預覽 (Hard Prompts):")
    preview_count = 0
    for k, v in descriptions.items():
        if preview_count < 5:
            print(f"[{k}]: a photo of a {k}, {v}")
            preview_count += 1

if __name__ == '__main__':
    main()
