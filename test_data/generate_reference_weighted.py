import os
import random
import pandas as pd

def create_reference_csv(n_dir: str, afib_dir: str, output_file: str, max_total: int = 10000):
    # 获取所有的 .npy 文件
    n_files = [f for f in os.listdir(n_dir) if f.endswith('.npy')]
    afib_files = [f for f in os.listdir(afib_dir) if f.endswith('.npy')]

    # 将所有文件及其标签汇总到一个列表
    files_with_labels = [(f, 'N') for f in n_files] + [(f, 'AFIB') for f in afib_files]

    # 如果总数超过 max_total, 根据标签的比例进行抽样
    total_files = len(n_files) + len(afib_files)

    if total_files > max_total:
        # 计算 N 和 AFIB 的比例
        n_ratio = len(n_files) / total_files
        afib_ratio = len(afib_files) / total_files

        # 根据比例计算需要抽取的样本数量
        n_sample_count = int(max_total * n_ratio)
        afib_sample_count = int(max_total * afib_ratio)

        # 按照计算好的数量进行抽样
        n_files = random.sample(n_files, n_sample_count)
        afib_files = random.sample(afib_files, afib_sample_count)

        # 重新汇总抽样后的文件列表
        files_with_labels = [(f, 'N') for f in n_files] + [(f, 'AFIB') for f in afib_files]

    # 打乱所有片段的顺序
    random.shuffle(files_with_labels)

    # 创建 DataFrame，文件名和标签
    df = pd.DataFrame(files_with_labels, columns=['file_name', 'label'])

    # 保存为 CSV 文件
    df.to_csv(output_file, index=False, header=False)
    print(f"REFERENCE.csv created at {output_file}")

if __name__ == "__main__":
    n_dir = "N"
    afib_dir = "AFIB"
    output_file = "REFERENCE.csv"
    create_reference_csv(n_dir, afib_dir, output_file)
