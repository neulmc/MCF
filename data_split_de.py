import os
import shutil
import random
from pathlib import Path

def split_dataset(root_path, train_ratio=0.75, random_seed=42):
    """
    将多模态数据集划分为训练集和测试集

    Args:
        root_path: 数据集根目录 (E:\multi-modal\full\train)
        train_ratio: 训练集比例 (默认1500/2000=0.75)
        random_seed: 随机种子，保证结果可重现
    """

    # 设置随机种子
    random.seed(random_seed)
    out_root_path = r"E:\M3FD\ei"
    # 定义模态文件夹名称
    modalities = ['visible', 'infrared', 'labels']

    # 原始数据路径
    original_paths = {
        'visible': os.path.join(root_path, 'visible'),
        'infrared': os.path.join(root_path, 'infrared'),
        'labels': os.path.join(root_path, 'labels')
    }

    # 检查所有文件夹是否存在
    for modality, path in original_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件夹: {path}")

    # 获取所有图像文件名（假设所有模态的图像名称相同）
    # 从visible文件夹获取所有图片文件名
    visible_files = [f for f in os.listdir(original_paths['visible'])
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    total_images = len(visible_files)
    print(f"总共找到 {total_images} 张图片")

    # 计算训练集和测试集数量
    train_count = int(total_images * train_ratio)
    test_count = total_images - train_count

    print(f"训练集: {train_count} 张")
    print(f"测试集: {test_count} 张")

    # 随机打乱文件名
    random.shuffle(visible_files)

    # 划分文件名
    train_files = visible_files[:train_count]
    test_files = visible_files[train_count:]

    # 创建新的目录结构
    # 新根目录（可以修改为你想要的输出路径）
    output_root = out_root_path  # 这里设置为E:\multi-modal\full
    # 或者可以创建一个新的文件夹
    # output_root = os.path.join(root_path, '..', 'split_dataset')

    for modality in modalities:
        # 创建模态对应的目录
        modality_path = os.path.join(output_root, modality)
        train_path = os.path.join(modality_path, 'train')
        test_path = os.path.join(modality_path, 'test')

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        print(f"创建目录: {modality_path}")

    # 复制文件到新目录
    print("\n开始复制文件...")

    for modality in modalities:
        print(f"\n处理 {modality} 模态...")
        src_dir = original_paths[modality]

        # 复制训练集文件
        train_dst = os.path.join(output_root, modality, 'train')
        for file_name in train_files:
            src_file = os.path.join(src_dir, file_name)
            dst_file = os.path.join(train_dst, file_name)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                shutil.copy2(src_file.split('.')[0] + '.txt', dst_file.split('.')[0] + '.txt')
                print(f"警告: 文件不存在 {src_file}")

        print(f"  训练集复制完成: {len(train_files)} 张")

        # 复制测试集文件
        test_dst = os.path.join(output_root, modality, 'test')
        for file_name in test_files:
            src_file = os.path.join(src_dir, file_name)
            dst_file = os.path.join(test_dst, file_name)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                shutil.copy2(src_file.split('.')[0] + '.txt', dst_file.split('.')[0] + '.txt')
                print(f"警告: 文件不存在 {src_file}")

        print(f"  测试集复制完成: {len(test_files)} 张")

    # 保存划分信息到文本文件
    split_info_path = os.path.join(output_root, 'dataset_split_info.txt')
    with open(split_info_path, 'w', encoding='utf-8') as f:
        f.write(f"数据集划分信息\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"训练集数量: {train_count}\n")
        f.write(f"测试集数量: {test_count}\n")
        f.write(f"随机种子: {random_seed}\n")
        f.write(f"\n训练集文件列表:\n")
        for file_name in train_files:
            f.write(f"  {file_name}\n")
        f.write(f"\n测试集文件列表:\n")
        for file_name in test_files:
            f.write(f"  {file_name}\n")

    print(f"\n{'=' * 50}")
    print(f"数据集划分完成！")
    print(f"新数据集保存在: {output_root}")
    print(f"划分信息保存在: {split_info_path}")

    # 验证各模态文件数量是否一致
    print(f"\n验证结果:")
    for modality in modalities:
        train_path = os.path.join(output_root, modality, 'train')
        test_path = os.path.join(output_root, modality, 'test')

        train_count_actual = len(
            [f for f in os.listdir(train_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        test_count_actual = len(
            [f for f in os.listdir(test_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

        print(f"{modality}: 训练集={train_count_actual}, 测试集={test_count_actual}")


def main():
    # 设置路径
    root_path = r"E:\M3FD\full"

    # 执行划分
    # 可以调整训练集比例，例如0.75表示75%训练，25%测试
    split_dataset(root_path, train_ratio=0.80, random_seed=42)


if __name__ == "__main__":
    main()