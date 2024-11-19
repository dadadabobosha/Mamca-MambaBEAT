import os
import shutil
import numpy as np
from tqdm import tqdm
import random


def get_patient_id_from_filename(filename):
    # 提取文件名中病人编号的函数，假设病人编号是文件名的第一部分
    return filename.split('_')[0]


def sample_patient_files(input_dir, output_dir, num_patients=20, num_files_per_patient=5):
    for subfolder in ["N", "AFIB"]:
        input_subfolder = os.path.join(input_dir, subfolder)
        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # 获取文件列表并分组为 {patient_id: [file1, file2, ...]}
        patient_files = {}
        for filename in os.listdir(input_subfolder):
            if filename.endswith(".npy"):
                patient_id = get_patient_id_from_filename(filename)
                if patient_id not in patient_files:
                    patient_files[patient_id] = []
                patient_files[patient_id].append(filename)

        # 随机选取 num_patients 个病人
        selected_patients = random.sample(list(patient_files.keys()), num_patients)

        # 对每个病人随机选取 num_files_per_patient 个文件并复制
        with tqdm(total=num_patients * num_files_per_patient, desc=f"Copying files for {subfolder}",
                  unit="file") as pbar:
            for patient_id in selected_patients:
                files = patient_files[patient_id]

                # 如果文件数量不足，取全部文件；否则随机选取 num_files_per_patient 个文件
                if len(files) < num_files_per_patient:
                    selected_files = files
                else:
                    selected_files = random.sample(files, num_files_per_patient)

                for filename in selected_files:
                    src_file = os.path.join(input_subfolder, filename)
                    dst_file = os.path.join(output_subfolder, filename)
                    shutil.copy(src_file, dst_file)
                    pbar.update(1)


if __name__ == "__main__":
    input_directory = "/work/scratch/js54mumy/icentia11k/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0/seg_npy4/10k/"
    output_directory = "./10k/"

    # 执行文件复制操作
    sample_patient_files(input_directory, output_directory)
