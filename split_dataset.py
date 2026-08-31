import os
import shutil
import random

# ================= 配置参数 =================
# 原始数据集路径 (包含 n01532829 等子文件夹)
SOURCE_DIR = "mini-imagenet/images" 
# 划分后的目标数据集路径
TARGET_DIR = "mini-imagenet"

# 划分数量
TRAIN_NUM = 20
VAL_NUM = 30
# 剩余的全部作为测试集 (正常情况下是 550)

# 固定随机种子，保证每次划分的图片完全一样，方便实验复现
RANDOM_SEED = 42
# ============================================

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # 设置随机种子
    random.seed(RANDOM_SEED)

    # 创建目标主文件夹和子文件夹
    for split in ['train', 'val', 'test']:
        create_dir_if_not_exists(os.path.join(TARGET_DIR, split))

    # 获取所有类别文件夹 (过滤掉非文件夹类型的文件)
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    classes.sort() # 排序以保证一致性

    print(f"找到 {len(classes)} 个类别文件夹，开始划分数据...")

    for i, cls_name in enumerate(classes):
        cls_path = os.path.join(SOURCE_DIR, cls_name)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 排序后再打乱，消除操作系统读取顺序带来的随机性差异
        images.sort()
        random.shuffle(images)

        # 检查数量是否足够
        total_images = len(images)
        if total_images < (TRAIN_NUM + VAL_NUM):
            print(f"⚠️ 警告: 类别 {cls_name} 的图片数量 ({total_images}) 不足划分所需！跳过该类。")
            continue

        # 划分列表
        train_imgs = images[:TRAIN_NUM]
        val_imgs = images[TRAIN_NUM : TRAIN_NUM + VAL_NUM]
        test_imgs = images[TRAIN_NUM + VAL_NUM :]

        # 为当前类别在 train, val, test 下创建子文件夹
        for split in ['train', 'val', 'test']:
            create_dir_if_not_exists(os.path.join(TARGET_DIR, split, cls_name))

        # 复制文件函数
        def copy_files(img_list, split_name):
            for img in img_list:
                src = os.path.join(cls_path, img)
                dst = os.path.join(TARGET_DIR, split_name, cls_name, img)
                shutil.copy2(src, dst) # copy2 会保留文件的元数据

        # 执行复制
        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')

        # 打印进度
        if (i + 1) % 10 == 0 or (i + 1) == len(classes):
            print(f"进度: {i + 1}/{len(classes)} 类别处理完毕...")

    print("\n✅ 数据集划分完成！")
    print(f"已保存至: {TARGET_DIR}")
    print(f"检查你的目录结构：\n{TARGET_DIR}/\n  ├── train/ (每类 {TRAIN_NUM} 张)\n  ├── val/   (每类 {VAL_NUM} 张)\n  └── test/  (每类剩余图片，约 550 张)")

if __name__ == '__main__':
    main()