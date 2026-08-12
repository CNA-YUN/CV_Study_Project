import zipfile
import os
from tqdm import tqdm  # 导入进度条库
from c2net.context import prepare, upload_output

c2net_context = prepare()

source_zip = c2net_context.dataset_path + "/OxfordPet" + "/oxford-iiit-pet.zip"
target_dir = c2net_context.output_path
# os.makedirs(target_dir, exist_ok=True)

with zipfile.ZipFile(source_zip, 'r') as zip_ref:
    file_list = zip_ref.infolist()

    # tqdm 自动显示进度百分比、速度和剩余时间
    for file_info in tqdm(file_list, desc="解压中", unit="个文件"):
        zip_ref.extract(file_info, path=target_dir)

print("全部解压完成！")
# 回传结果到openi，只有训练任务才能回传
upload_output()
