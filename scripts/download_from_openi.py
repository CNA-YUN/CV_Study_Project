# 从 OpenI 下载数据集或模型
import openi
from openi import openi_download_file
from _init_ import DATA_DIR
openi.login(token='0ec626ba6fdac57e828ec0b3bb67601b544b2483')

openi_download_file("ppsuser/msd_Task09_Spleen", repo_type="dataset", local_dir=DATA_DIR+"/msd_Task09_Spleen", max_workers=10)
