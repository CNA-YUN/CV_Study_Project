``` python
'''
从子集中按随机种子 42 抽取 200 张；resize 到 224×224；同时保存 RGB 版本、灰度版本和直方图均衡化版本，1 张 4×5 对比拼图。

parameters:
 --input_csv:输入 CSV 文件路径
 --save_dir:输出根目录
 --image_size:resize 后的图像尺寸
 --seed:随机种子
'''
python preprocess_cifar10.py --input_csv cifar10_inventory.csv --save_dir outputs/m1_task3_preprocess --image_size 224 --seed 42
```